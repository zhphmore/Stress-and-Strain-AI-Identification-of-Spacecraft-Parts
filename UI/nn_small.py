import numpy as np
import pandas as pd
import torch

import small_A_model
import small_B_model

num_each_region_small = [1016, 1034, 1009, 1028]
num_region_row_small = [0, 1016, 2050, 3059]
num_each_region_core_small = [112, 127, 115, 127]
num_region_row_core_small = [0, 112, 239, 354]


# ********************************************************************************
# ****                                                                        ****
# ****                            小型零件  神经网络A                            ****
# ****                                                                        ****
# ********************************************************************************
def preprocess_small_A(data_clip):
    data_input = data_clip * 0.1
    return data_input


def postprocess_small_A(data_output):
    data_FM = np.zeros_like(data_output)
    data_FM[0] = (data_output[0] - 0.5) * 120
    data_FM[1] = (data_output[1] - 0.5) * 120
    data_FM[2] = (data_output[2] - 0.5) * 400
    data_FM[3] = (data_output[3] - 0.5) * 2
    data_FM[4] = (data_output[4] - 0.5) * 2
    data_FM[5] = (data_output[5] - 0.5) * 4
    return data_FM


def nn_small_A(data_clip):
    data_input = torch.tensor(preprocess_small_A(data_clip))

    name_pth = 'small_A_1.pth'

    loaded_model = small_A_model.NN_FM()
    loaded_model.load_state_dict(torch.load(name_pth, weights_only=True, map_location='cpu'))

    loaded_model.eval()

    with torch.no_grad():
        data_output = loaded_model(data_input).numpy()

    data_FM = postprocess_small_A(data_output)

    return data_FM


# ********************************************************************************
# ****                                                                        ****
# ****                            小型零件  神经网络B                            ****
# ****                                                                        ****
# ********************************************************************************
def preprocess_small_B(data_FM):
    data_input = np.zeros_like(data_FM)
    data_input[0] = data_FM[0] / 120 + 0.5
    data_input[1] = data_FM[1] / 120 + 0.5
    data_input[2] = data_FM[2] / 2000 + 0.5
    data_input[3] = data_FM[3] / 2 + 0.5
    data_input[4] = data_FM[4] / 2 + 0.5
    data_input[5] = data_FM[5] / 4 + 0.5
    return data_input


def postprocess_small_B(data_output_S, data_output_E):
    data_output_S *= 1000
    data_output_E /= 1000
    return data_output_S, data_output_E


def postprocess_small_B_S(data_output_S):
    data_output_S *= 1000
    return data_output_S


def nn_small_B_region(data_input, target_region_id):
    target_region_id = int(target_region_id)

    name_pth_S = 'small_B_S_R' + str(target_region_id) + '_1.pth'

    loaded_model_S = small_B_model.NN_S(target_region_id)
    loaded_model_S.load_state_dict(torch.load(name_pth_S, weights_only=True, map_location='cpu'))

    loaded_model_S.eval()

    with torch.no_grad():
        data_output_S = loaded_model_S(data_input).numpy()

    outputs_S = postprocess_small_B_S(data_output_S)

    return outputs_S


def nn_small_B_region_core(data_input, target_region_id):
    target_region_id = int(target_region_id)

    name_pth_S = 'small_B_S_core_top5_R' + str(target_region_id) + '_1.pth'
    name_pth_E = 'small_B_E_core_top5_R' + str(target_region_id) + '_1.pth'

    loaded_model_S = small_B_model.NN_S_core(target_region_id)
    loaded_model_S.load_state_dict(torch.load(name_pth_S, weights_only=True, map_location='cpu'))
    loaded_model_E = small_B_model.NN_E_core(target_region_id)
    loaded_model_E.load_state_dict(torch.load(name_pth_E, weights_only=True, map_location='cpu'))

    loaded_model_S.eval()
    loaded_model_E.eval()

    with torch.no_grad():
        data_output_S = loaded_model_S(data_input).numpy()
        data_output_E = loaded_model_E(data_input).numpy()

    outputs_S, outputs_E = postprocess_small_B(data_output_S, data_output_E)

    ans = np.zeros(8)

    id_argmax = np.argmax(outputs_S)

    ans[0] = int(num_region_row_core_small[int(target_region_id - 1)] + id_argmax)
    ans[1] = outputs_S[id_argmax]
    ans[2:8] = outputs_E[int(6 * id_argmax):int(6 * id_argmax + 6)] * 1000000

    return ans


def nn_small_B(data_FM):
    data_input = torch.tensor(preprocess_small_B(data_FM))

    data_region_S = np.zeros(4087)

    file_region_core_top5_elem = 'region_core_top5_elem_small.csv'
    data_region_core_top5_elem = pd.read_csv(file_region_core_top5_elem, header=None, encoding="utf-8").to_numpy()

    ans_IDSE = np.zeros(8)

    for i in range(4):
        outputs_S = nn_small_B_region(data_input, int(i + 1))

        data_region_S[int(num_region_row_small[i]):int(num_region_row_small[i] + num_each_region_small[i])] = outputs_S

        ans_i = nn_small_B_region_core(data_input, int(i + 1))

        if ans_i[1] > ans_IDSE[1]:
            ans_IDSE[0] = int(data_region_core_top5_elem[int(ans_i[0])][0] - 1)
            ans_IDSE[1:8] = ans_i[1:8]

    return data_region_S, ans_IDSE
