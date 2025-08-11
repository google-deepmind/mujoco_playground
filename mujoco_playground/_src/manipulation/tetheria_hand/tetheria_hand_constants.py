# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for tetheria hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "tetheria_hand"
CUBE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube.xml"

NQ = 20
NV = 20
NU = 20

JOINT_NAMES = [
    # index
    "right_index_mcp_flex",
    "right_index_mcp_abd",
    "right_index_pip",
    "right_index_dip",
    # "if_dip",
    # middle
    "right_middle_mcp_flex",
    "right_middle_mcp_abd",
    "right_middle_pip",
    "right_middle_dip",
    # "mf_dip",
    # ring
    "right_ring_mcp_flex",
    "right_ring_mcp_abd",
    "right_ring_pip",
    "right_ring_dip",
    # "rf_dip",
    # pinky
    "right_pinky_mcp_flex",
    "right_pinky_mcp_abd",
    "right_pinky_pip",
    "right_pinky_dip",
    # "th_dip",
    # thumb
    "right_thumb_cmc_flex",
    "right_thumb_cmc_abd",
    "right_thumb_mcp",
    "right_thumb_ip",
    # "th_ipl",
]

# CONTROLJOINT_NAMES = [
#     # index
#     "right_index_mcp_flex",
#     "right_index_mcp_abd",
#     "right_index_pip",
#     # "right_index_dip",
#     # "if_dip",
#     # middle
#     "right_middle_mcp_flex",
#     "right_middle_mcp_abd",
#     "right_middle_pip",
#     # "right_middle_dip",
#     # "mf_dip",
#     # ring
#     "right_ring_mcp_flex",
#     "right_ring_mcp_abd",
#     "right_ring_pip",
#     # "right_ring_dip",
#     # "rf_dip",
#     # pinky
#     "right_pinky_mcp_flex",
#     "right_pinky_mcp_abd",
#     "right_pinky_pip",
#     # "right_pinky_dip",
#     # "th_dip",
#     # thumb
#     "right_thumb_cmc_flex",
#     "right_thumb_cmc_abd",
#     # "right_thumb_mcp",
#     "right_thumb_ip",
#     # "th_ipl",
# ]

# ENFORCE_JOINT_PAIRS = [
#     ("right_index_pip", "right_index_dip"),
#     ("right_middle_pip", "right_middle_dip"),
#     ("right_ring_pip", "right_ring_dip"),
#     ("right_pinky_pip", "right_pinky_dip"),
#     ("right_thumb_ip", "right_thumb_mcp"),
# ]

ACTUATOR_NAMES = [
    # index
    "right_index_A_mcp_flex",
    "right_index_A_mcp_abd",
    "right_index_A_pip",
    "right_index_A_dip",
    # "if_dip_act",
    # middle
    "right_middle_A_mcp_flex",
    "right_middle_A_mcp_abd",
    "right_middle_A_pip",
    "right_middle_A_dip",
    # "mf_dip_act",
    # ring
    "right_ring_A_mcp_flex",
    "right_ring_A_mcp_abd",
    "right_ring_A_pip",
    "right_ring_A_dip",
    # "rf_dip_act",
    # pinky
    "right_pinky_A_mcp_flex",
    "right_pinky_A_mcp_abd",
    "right_pinky_A_pip",
    "right_pinky_A_dip",
    # "th_dip_act",
    # thumb
    "right_thumb_A_cmc_flex",
    "right_thumb_A_cmc_abd",
    "right_thumb_A_pip",
    "right_thumb_A_dip",
    # "th_ipl_act",
    # "th_mcp_act",
    # "th_ipl_act",
]

FINGERTIP_NAMES = [
    "th_tip",
    "if_tip",
    "mf_tip",
    "rf_tip",
    "pf_tip",
]
