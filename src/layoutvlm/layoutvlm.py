import os
import json
import re
import numpy as np
import torch
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from .sandbox import SandBoxEnv
from tqdm import tqdm
from typing import List, Dict, Literal, Optional
from utils.plot_utils import load_image, overlay_bounding_box
import base64
import collections
from utils.placement_utils import get_random_placement
from prompts.layoutvlm import base_prompt
from utils.blender_render import render_existing_scene
from utils.blender_utils import reset_blender
from collections import OrderedDict
import prompts.layoutvlm.short_prompt as short_prompt
import imageio
from PIL import Image
from utils.placement_utils import replace_z_rot_degree_to_rpy_radians


def extract_python_program(input_text):
    # Accept ```python, ```py, or bare ``` fences, with optional whitespace.
    for pattern in (r"```(?:python|py)\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        matches = re.findall(pattern, input_text, flags=re.DOTALL)
        if matches:
            return [m.strip() for m in matches]
    return []


def _strip_code_fences(text):
    """Remove stray leading/trailing markdown fence lines from code text."""
    lines = text.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

def extract_description_program(input_text):
    pattern = r"\*\*\*(.*?)\*\*\*"
    matches = re.findall(pattern, input_text, flags=re.DOTALL)
    return matches

def extract_json(input_text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, input_text, flags=re.DOTALL)
    return matches

class LayoutVLM:

    def __init__(self, save_dir, gpt_4o_model_name="gpt-4o", asset_source="objaverse", mode="finetuned", visual_mark_mode="new_coord",
                 ft_original_model_id=None, ft_model_checkpoint=None, convert_z_rot_degree_to_rpy_radians=True, max_place_remaining_retry=2,
                 numerical_value_only=False):
        # initialize llm
        self.mode = mode
        self.asset_source = asset_source
        self.save_dir = save_dir
        # Route through an OpenAI-compatible endpoint (chatanywhere by default)
        # and standardise on a single pinned model for the audit. Override via
        # VLMUNR_LLM_MODEL / OPENAI_BASE_URL.
        import os as _os
        _model = _os.environ.get("VLMUNR_LLM_MODEL", "gpt-5.1-2025-11-13")
        _base = _os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
        _key = _os.environ.get("OPENAI_API_KEY")
        _kw = dict(max_tokens=16384, base_url=_base, api_key=_key)
        self.llm_slow = ChatOpenAI(model_name=_model, **_kw)
        self.llm_slow_mini = ChatOpenAI(model_name=_model, **_kw)
        self.llm_slow_grouping = ChatOpenAI(model_name=_model, **_kw)
        self.visual_mark_mode = visual_mark_mode
        self.numerical_value_only = numerical_value_only

        self.ft_original_model_id = ft_original_model_id
        self.ft_model_checkpoint = ft_model_checkpoint
        self.convert_z_rot_degree_to_rpy_radians = convert_z_rot_degree_to_rpy_radians
        self.max_place_remaining_retry = max_place_remaining_retry
        


    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def mark_image(self, visual_marks, input_path, output_path, image_size=(512, 512)):
        if self.visual_mark_mode == "grid":
            assert False
        elif self.visual_mark_mode == "old_coord":
            assert False
        elif self.visual_mark_mode == "new_coord":
            assert input_path is not None
            from utils.plot_utils import annotate_image_with_coordinates
            annotate_image_with_coordinates(input_path, visual_marks, output_path, default_color='red')
        else:
            raise NotImplementedError(f"Visual mark mode {self.visual_mark_mode} not implemented")


    def get_asset_groups(self, task, MAX_ATTEMPTS=5, save_dir=None, include_position_in_prompt=False):
        ### old version with Pydantic
        #class Assets(BaseModel):
        #    assets: List[str] = Field(description="the list of of asset names in the group")
        #    layout_criteria: str = Field(description="the layout instruction for this group of assets describing how they should be placed in the 3D scene. This instruction should mostly pertain to the assets in this group.")
        #class AssetGroups(BaseModel):
        #    group: List[Assets] = Field(description="List of grouped assets that should be placed together")
        # parser = PydanticOutputParser(pydantic_object=AssetGroups)
        # prompt = open("prompts/layoutvlm/unused/asset_grouping.txt", "r").read()
        #prompt = prompt.replace("TASK_DESCRIPTION", task["task_description"])
        # prompt.replace("OBJECT_LIST", object_list)
        #prompt = PromptTemplate(
        #    template=prompt + "\n{format_instructions}\n",
        #    input_variables=["asset_lists"],
        #    partial_variables={"format_instructions": parser.get_format_instructions()},
        #)
        #chain = prompt | self.llm_fast
        #result = chain.invoke(input={"asset_lists": object_list})
        from prompts.layoutvlm import grouping
        object_list = "[asset name] | [description] | [bounding box] \n | [position] \n" 
        for instance_id, asset in task["assets"].items():
            if include_position_in_prompt:
                object_list += f'{instance_id} | {asset["annotations"]["description"]} | {asset["assetMetadata"]["boundingBox"]} | {asset["annotations"]["position"]}\n'
            else:
                object_list += f'{instance_id} | {asset["annotations"]["description"]} | {asset["assetMetadata"]["boundingBox"]}\n'

        prompt = grouping.grouping_v1_flat_text
        prompt = prompt.replace("ASSET_LISTS", object_list)
        prompt = prompt.replace("TASK_DESCRIPTION", task["task_description"])
        prompt = prompt.replace("LAYOUT_CRITERIA", task["layout_criteria"])
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
            ]
        )

        for attempt_idx in range(MAX_ATTEMPTS):
            try:
                response = self.llm_slow_grouping.invoke([message])
                if save_dir is not None:
                    with open(f"{save_dir}/grouping_{attempt_idx}.txt", "w") as f:
                        f.write(prompt + "\n\n" + response.content)
                matches = extract_json(response.content)
                result = json.loads(matches[-1])
                return result["list"]
            except Exception as e:
                print("Retrying in get_asset_groups ...", e)



    def get_initialization(self, final_prompt, encoded_image=None):
        if encoded_image is None:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt},
                ]
            )
        else:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            )
        response = self.llm_slow_mini.invoke([message])
        response_text = response.content
        matches = extract_python_program(response_text)
        if matches:
            program = matches[0]
        else:
            program = ""
        return program

    def prepare_finetuned_vlm_prompt(self, final_prompt, encoded_images=[]):
        conversation = []
        user_content = [
            {
                "type": "text",
                "text": final_prompt.replace("<image>", "")
            }
        ]
        user_content.extend([{"type": "image"} for _ in range(len(encoded_images))])
        conversation.append({
            "role": "user",
            "content": user_content
        })
        image_list = [Image.open(image_path).convert("RGB") for image_path in encoded_images]
        return conversation, image_list

    def get_constraint_program(self, final_prompt, current_scene_image_path_dict, current_group_asset_img_path_dict, program_save_path=None):
        image_paths = []
        if self.mode != "no_image":
            top_down_scene_image_path = current_scene_image_path_dict["top_down_rendering"]
            side_scene_image_path = current_scene_image_path_dict["side_rendering_45_3"]
            asset_images = [current_group_asset_img_path_dict[asset_name] for asset_name in current_group_asset_img_path_dict]
            image_paths = [top_down_scene_image_path, side_scene_image_path] + asset_images
        

        messages = [
            (
                "system",
                "You are a coding agent. PLEASE DO NOT REPEAT ANY OF THE CODE CODE GIVEN. DO NOT RE-INITIALIZE ANY OF THE GIVEN VARIABLES."
            )
        ]
        content = [{"type": "text", "text": final_prompt}]

        encoded_images = [self.encode_image(image_path) for image_path in image_paths]
        for encoded_image in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                }
            )
        message = HumanMessage(content=content)
        messages.append(message)
        response = self.llm_slow.invoke(messages)
        response_text = response.content
        if self.convert_z_rot_degree_to_rpy_radians:
            response_text = replace_z_rot_degree_to_rpy_radians(response_text)

        with open(program_save_path, "w") as f:
            f.write(response_text)
        matches = extract_python_program(response_text)
        if matches:
            constraint_program = matches[0]
        else:
            # No clean fenced block; strip any stray fence lines from the raw text.
            constraint_program = _strip_code_fences(response_text)

        ### remove re-initialized variables
        matches = list(re.finditer(r"\w+ = Assets\(", constraint_program))
        if matches:
            last_match = matches[-1]
            # Extract the code after the last mat    
            end_of_line = constraint_program.find("\n", last_match.end())
            # Extract the code after the last match's line
            constraint_program = constraint_program[end_of_line + 1:].strip()
            # matches = list(re.finditer(r"\w+ = Assets\(", constraint_program))
        return constraint_program

    @staticmethod
    def get_task_program(grouped_assets, task, verify_asset_var_name_to_count=None):
        """
        Args:
            grouped_assets: list of grouped assets
            task: input json of the scene and the assets
            verify_asset_var_name_to_count: used to verify whether count is correct
        """
        program = "# Walls that define the boundary of the scene\n"
        floor_vertices = task['boundary']['floor_vertices']
        num_walls = len(floor_vertices)
        program += "walls = [\n"
        for wall_idx in range(len(task["boundary"]["floor_vertices"])):
            size_str1 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[wall_idx][0],floor_vertices[wall_idx][1], floor_vertices[wall_idx][2]
            )
            size_str2 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[(wall_idx+1)%num_walls][0],floor_vertices[(wall_idx+1)%num_walls][1], floor_vertices[(wall_idx+1)%num_walls][2]
            )
            if wall_idx == len(task["boundary"]["floor_vertices"]) - 1:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2})\n]\n"
            else:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2}),\n"

        program += "\n# Existing assets placed in the scene:\n"
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            if instance_uid in grouped_assets:
                continue
            #printasset)
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            if verify_asset_var_name_to_count is not None:
                assert value['count'] == verify_asset_var_name_to_count[asset['asset_var_name']], (
                    "value['count'] ({}) != verify_asset_var_name_to_count[asset['asset_var_name']] ({}) for {}".format(
                        value['count'], verify_asset_var_name_to_count[asset['asset_var_name']], asset['asset_var_name']))
            program += (f"{re.sub(r'[^A-Za-z0-9_]', '_', asset['asset_var_name'])} = Assets("
                f"description={(asset['description'] or '')!r}, "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )

        program += '\n# New assets to be placed\n'
        ### FORM for loops
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            if instance_uid not in grouped_assets:
                continue
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            if verify_asset_var_name_to_count is not None:
                assert value['count'] == verify_asset_var_name_to_count[asset['asset_var_name']], (
                    "value['count'] ({}) != verify_asset_var_name_to_count[asset['asset_var_name']] ({}) for {}".format(
                        value['count'], verify_asset_var_name_to_count[asset['asset_var_name']], asset['asset_var_name']))
            program += (f"{re.sub(r'[^A-Za-z0-9_]', '_', asset['asset_var_name'])} = Assets("
                f"description={(asset['description'] or '')!r}, "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )
        return program

    def _solve_single_group(self, task, layout_criteria, placed_assets, group_assets, _save_dir, 
                            include_image=True, MAX_ATTEMPTS=5, only_initialize=False):
        #############################################################################
        ### prepare scene images
        #############################################################################
        current_scene_image_path_dict = {}
        if include_image:
            if self.mode == "no_visual_coordinate":
                output_images, visual_marks = render_existing_scene(
                    placed_assets, task, save_dir=_save_dir,
                    high_res=True, render_top_down=True,
                    # For 3d front .obj
                    apply_3dfront_texture=True, combine_obj_components=True,
                    # this constant is used when generating training data. see front3d_v2.py
                    fov_multiplier=1.3,
                    # Disable all annotation flags
                    add_coordinate_mark=False,
                    annotate_object=True,
                    annotate_wall=True,
                    add_object_bbox=False
                )
            elif self.mode == "no_visual_assetname":
                output_images, visual_marks = render_existing_scene(
                    placed_assets, task, save_dir=_save_dir,
                    high_res=True, render_top_down=True,
                    # For 3d front .obj
                    apply_3dfront_texture=True, combine_obj_components=True,
                    # this constant is used when generating training data. see front3d_v2.py
                    fov_multiplier=1.3,
                    # Disable all annotation flags
                    add_coordinate_mark=True,
                    annotate_object=False,
                    annotate_wall=False,
                    add_object_bbox=False
                )
            elif self.mode == "no_visual_mark":
                output_images, visual_marks = render_existing_scene(
                    placed_assets, task, save_dir=_save_dir,
                    high_res=True, render_top_down=True,
                    # For 3d front .obj
                    apply_3dfront_texture=True, combine_obj_components=True,
                    # this constant is used when generating training data. see front3d_v2.py
                    fov_multiplier=1.3,
                    # Disable all annotation flags
                    add_coordinate_mark=False,
                    annotate_object=False,
                    annotate_wall=False,
                    add_object_bbox=False
                )
            else:
                output_images, visual_marks = render_existing_scene(
                    placed_assets, task, save_dir=_save_dir,
                    high_res=True, render_top_down=True,
                    # For 3d front .obj
                    apply_3dfront_texture=True, combine_obj_components=True,
                    # this constant is used when generating training data. see front3d_v2.py
                    fov_multiplier=1.3
                )
            reset_blender()
            ### render the incomplete scene with the visual marks
            for _image in output_images:
                current_scene_image_path_dict[os.path.basename(_image).split('.')[0]] = _image

        #############################################################################
        ### prepare asset images (omitted)
        #############################################################################
        current_group_asset_img_path_dict = OrderedDict()

        #############################################################################
        ### form the prompt get the list of asset names in the group
        #############################################################################
        _task = task.copy()
        _task["assets"] = {k: v for k, v in task["assets"].items() if (k in placed_assets.keys() or k in group_assets)}
        task_program_for_prompt = self.get_task_program(group_assets, _task)

        asset_placed_list = [asset_name.replace('-', '[') + ']' if '-' in asset_name else asset_name for asset_name in task['assets'].keys() if asset_name not in group_assets]
        asset_be_placed_list = [asset_name.replace('-', '[') + ']' if '-' in asset_name else asset_name for asset_name in group_assets]
        #pattern = re.compile(r"(\b\w+)\[(\d+)\]")
        def split_asset_string(asset):
            # Regex to match the pattern "{asset_type}[{index}]"
            match = re.match(r"(\w+)\[(\d+)\]", asset)
            if match:
                asset_type, index = match.groups()
                return asset_type, int(index)
            else:
                return None, None
        index_map = {}
        for asset in asset_be_placed_list:
            asset_type, index = split_asset_string(asset)
            if asset_type is None:
                print("Asset string format is incorrect: ", asset)
                continue
            if asset_type in index_map:
                index_map[asset_type] = min(index_map[asset_type], index)
            else:
                index_map[asset_type] = index
        # Step 2: Adjust indices to start from 0 for each asset type
        normalized_assets = []
        for asset in asset_be_placed_list:
            asset_type, index = split_asset_string(asset)
            if asset_type is None:
                print("Asset string format is incorrect: ", asset)
                continue
            # Subtract the minimum index found for this asset type to normalize
            new_index = int(index) - index_map[asset_type]
            normalized_assets.append(f"{asset_type}[{new_index}]")
        replacement_map = dict(zip(normalized_assets, asset_be_placed_list))
        # note: use the newest prompt
        # final_prompt = base_prompt.get_layout_prompt(task_program_for_prompt, layout_criteria)
        final_prompt = short_prompt.get_layout_prompt(task_program_for_prompt, {"layout_criteria": layout_criteria}, self.numerical_value_only)

        os.makedirs(_save_dir, exist_ok=True)
        with open(f"{_save_dir}/prompt.txt", "w") as f:
            f.write(final_prompt)
            #f.write(f"Asset_index: {', '.join(asset_be_placed_list)}")


        for attempt_idx in range(MAX_ATTEMPTS):
            try:
                # clear constraints
                self.sandbox.execute_code("solver.constraints = []\n")
                save_path = f"{_save_dir}/llm_output_program_{attempt_idx}.py"
                constraint_program = self.get_constraint_program(final_prompt,
                                                                 current_scene_image_path_dict,
                                                                 current_group_asset_img_path_dict,
                                                                 program_save_path=save_path)
                
                # print("BEFORE")
                # print(constraint_program)
                # Perform replacement
                for old_asset, new_asset in replacement_map.items():
                    constraint_program = constraint_program.replace(old_asset, new_asset)
                # print("AFTER")
                # print(constraint_program)
                #import pdb; pdb.set_trace()
                if constraint_program == "":
                    print("Constraint program is empty")
                    continue
                # find the last line of code with the pattern * = Assets(...)
                # get the constraint program only after the last line
                self.sandbox.sanity_check(group_assets, constraint_program)
                placed_assets = self.sandbox.solve(
                    placed_assets, group_assets, constraint_program, save_dir=_save_dir, only_initialize=only_initialize
                )
                break
            except Exception as e:
                print("Retrying ...", e)

        return placed_assets

    def solve(self, original_task, MAX_ATTEMPTS=8):
        """
        task is the input json of the scene and the assets 
        """
        task = original_task.copy()
        # sanitize asset_var_name so it is a valid Python identifier everywhere downstream
        import re as _re
        for _uid, _asset in task.get("assets", {}).items():
            if "asset_var_name" in _asset:
                _asset["asset_var_name"] = _re.sub(r"[^A-Za-z0-9_]", "_", _asset["asset_var_name"])
        #### initialize the sandbox and initialize all the variables
        self.sandbox = SandBoxEnv(task, mode=self.mode, save_dir=self.save_dir)
        task_program = self.get_task_program(list(task["assets"].keys()), task)
        self.sandbox.execute_code(base_prompt.CODE_FOR_SANDBOX + "\n" + task_program)
        self.sandbox.assign_instance_ids()
        self.sandbox.initialize_variables()
        self.sandbox.export_code()

        include_image = self.mode != "no_image"
        ### get asset groupings
        if self.mode == "one_shot":
            placed_assets = dict()
            unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
            num_groups = 0
            while len(unplaced_assets) and num_groups < 20:
                print(f"Placing unplaced assets -- group {num_groups}")
                _save_dir = os.path.join(self.save_dir, f"group_{num_groups}")
                placed_assets = self._solve_single_group(
                    task, task["layout_criteria"],
                    placed_assets, unplaced_assets, _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS
                )
                unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
                num_groups += 1

        else:
            group_list = self.get_asset_groups(task, save_dir=self.save_dir, MAX_ATTEMPTS=MAX_ATTEMPTS)
            with open(self.save_dir + "/grouping.json", "w") as f:
                json.dump(group_list, f, indent=4)

            placed_assets = self.sandbox.export_layout(incomplete_scene=True, use_degree=True)
            for group_idx, group in enumerate(group_list):
                _save_dir = os.path.join(self.save_dir, f"group_{group_idx}")
                os.makedirs(_save_dir, exist_ok=True)
                ### optionally: use another prompt to select the relevant context in the scene
                ### For now, we are using the top-down image plus the whole list of assets of the scene as the context

                layout_criteria = f"{task['layout_criteria']}. More specifically, Organize the {group['name']} of the room in the following way:\n"
                _key_relations_between_assets = "\n".join(group['key_relations_between_assets'])
                layout_criteria += f"{_key_relations_between_assets}\n"
                placed_assets = self._solve_single_group(
                    task, layout_criteria, placed_assets, group['assets'],
                    _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS,
                )
            # Find assets that are not placed yet
            num_groups = len(group_list)

        unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
        num_place_remaining_retry = 0
        # while not empty
        while len(unplaced_assets) and num_groups < 20 and num_place_remaining_retry < self.max_place_remaining_retry:
            print(f"Placing unplaced assets -- group {num_groups}")
            _save_dir = os.path.join(self.save_dir, f"group_{num_groups}")
            placed_assets = self._solve_single_group(
                task, task["layout_criteria"],
                placed_assets, unplaced_assets, _save_dir, include_image=include_image, MAX_ATTEMPTS=MAX_ATTEMPTS
            )
            unplaced_assets = set(task["assets"].keys()) - set(placed_assets.keys())
            num_groups += 1
            num_place_remaining_retry += 1

        if len(unplaced_assets) == 0:
            print("All assets have already been placed.")

        results = self.sandbox.export_layout(use_degree=True)
        ### save into one final gif
        if self.mode not in ["no_constraint", "finetuned"]:
            all_frames = []
            gif_files = []
            for group_idx in range(num_groups):
                if os.path.exists(f"{self.sandbox.save_dir}/group_{group_idx}/out.gif"):
                    gif_files.append(f"{self.sandbox.save_dir}/group_{group_idx}/out.gif")
            for gif_file in gif_files:
                gif = imageio.mimread(gif_file)  # Read all frames from the GIF
                all_frames.extend(gif)
            if len(all_frames) > 0:
                imageio.mimsave(f"{self.save_dir}/final.gif", all_frames)

        return results

    def get_simple_program(grouped_assets, task):
        """
        Args:
            grouped_assets: list of grouped assets
            task: input json of the scene and the assets
        """
        program = "# Walls that define the boundary of the scene\n"
        floor_vertices = task['boundary']['floor_vertices']
        num_walls = len(floor_vertices)
        program += "walls = [\n"
        for wall_idx in range(len(task["boundary"]["floor_vertices"])):
            size_str1 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[wall_idx][0],floor_vertices[wall_idx][1], floor_vertices[wall_idx][2]
            )
            size_str2 = "[{:.2f}, {:.2f}, {:.2f}]".format(
                floor_vertices[(wall_idx+1)%num_walls][0],floor_vertices[(wall_idx+1)%num_walls][1], floor_vertices[(wall_idx+1)%num_walls][2]
            )
            if wall_idx == len(task["boundary"]["floor_vertices"]) - 1:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2})\n]\n"
            else:
                program += f"    Wall(corner1={size_str1}, corner2={size_str2}),\n"

        ### FORM for loops
        new_asset_list  = []
        uid2asset = dict()
        for instance_uid, asset in task['assets'].items():
            print(instance_uid)
            new_asset_list.append(instance_uid)
            asset_uid = asset["asset_var_name"]
            if asset_uid not in uid2asset.keys():
                uid2asset[asset_uid] = {
                    "asset": asset,
                    "count": 1
                }
            else:
                uid2asset[asset_uid]["count"] += 1

        program += f"\n# New assets to be placed: [{', '.join(new_asset_list)}]\n"
        for asset_uid, value in uid2asset.items():
            asset = value["asset"]
            size_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
                asset['assetMetadata']['boundingBox']['x'],
                asset['assetMetadata']['boundingBox']['y'],
                asset['assetMetadata']['boundingBox']['z']
            )
            program += (f"{re.sub(r'[^A-Za-z0-9_]', '_', asset['asset_var_name'])} = Assets("
                f"description={(asset['description'] or '')!r}, "
                f"size={size_str}, "
                f"placements=[AssetInstance() for _ in range({value['count']})])\n"
            )
        
        return program
    

    def filter_constraint(self, image_path, final_prompt, save_path = None):
        # print(image_path)
        # print(final_prompt)
        with open(image_path, "rb") as image_file:
            encoded_image =  base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
                (
                    "system",
                    "You are a coding agent."
                )
            ]
        content = [{"type": "text", "text": final_prompt}]
        
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )
        message = HumanMessage(content=content)
        messages.append(message)

        for attempt_idx in range(3):
            try:
                response = self.llm_slow_mini.invoke(messages)
                response_text = response.content
                
                # print(response_text)
                # with open(save_path, "w") as f:
                #     f.write(response_text)
                constraint_program = extract_python_program(response_text)
                if constraint_program:
                    constraint_program = constraint_program[0]
                else:
                    constraint_program = ""

                task_description = extract_description_program(response_text)
                if task_description:
                    task_description = task_description[0]
                else:
                    task_description = ""
                #print(constraint_program)  
                
                return constraint_program, task_description
                
            except Exception as e:
                print("Retrying in filter_constraint ...", e)
