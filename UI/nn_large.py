import numpy as np
import pandas as pd
import torch

import large_A_model
import large_B_model

num_each_region_large = [735, 457, 453, 705, 730, 455, 455, 703]
num_region_row_large = [0, 735, 1192, 1645, 2350, 3080, 3535, 3990]
num_each_region_core_large = [125, 115, 117, 119, 126, 116, 118, 112]
num_region_row_core_large = [0, 125, 240, 357, 476, 602, 718, 836]


# ********************************************************************************
# ****                                                                        ****
# ****                            大型零件  神经网络A                            ****
# ****                                                                        ****
# ********************************************************************************
def preprocess_large_A(data_clip):
    data_input = data_clip * 0.1
    return data_input


def postprocess_large_A(data_output):
    data_FM = np.zeros_like(data_output)
    data_FM[0] = (data_output[0] - 0.5) * 320
    data_FM[1] = (data_output[1] - 0.5) * 320
    data_FM[2] = (data_output[2] - 0.5) * 1200
    data_FM[3] = (data_output[3] - 0.5) * 8
    data_FM[4] = (data_output[4] - 0.5) * 8
    data_FM[5] = (data_output[5] - 0.5) * 18
    return data_FM


def nn_large_A(data_clip):
    data_input = torch.tensor(preprocess_large_A(data_clip))

    name_pth = 'large_A_1.pth'

    loaded_model = large_A_model.NN_FM()
    loaded_model.load_state_dict(torch.load(name_pth, weights_only=True, map_location='cpu'))

    loaded_model.eval()

    with torch.no_grad():
        data_output = loaded_model(data_input).numpy()

    data_FM = postprocess_large_A(data_output)

    return data_FM


# ********************************************************************************
# ****                                                                        ****
# ****                            大型零件  神经网络B                            ****
# ****                                                                        ****
# ********************************************************************************
def preprocess_large_B(data_FM):
    data_input = np.zeros_like(data_FM)
    data_input[0] = data_FM[0] / 320 + 0.5
    data_input[1] = data_FM[1] / 320 + 0.5
    data_input[2] = data_FM[2] / 1200 + 0.5
    data_input[3] = data_FM[3] / 8 + 0.5
    data_input[4] = data_FM[4] / 8 + 0.5
    data_input[5] = data_FM[5] / 18 + 0.5
    return data_input


def postprocess_large_B(data_output_S, data_output_E):
    data_output_S *= 500
    data_output_E /= 1000
    return data_output_S, data_output_E


def postprocess_large_B_S(data_output_S):
    data_output_S *= 500
    return data_output_S


def nn_large_B_region(data_input, target_region_id):
    target_region_id = int(target_region_id)

    name_pth_S = 'large_B_S_R' + str(target_region_id) + '_1.pth'
    # name_pth_E = 'large_B_E_R' + str(target_region_id) + '_1.pth'

    loaded_model_S = large_B_model.NN_S(target_region_id)
    loaded_model_S.load_state_dict(torch.load(name_pth_S, weights_only=True, map_location='cpu'))
    # loaded_model_E = large_B_model.NN_E(target_region_id)
    # loaded_model_E.load_state_dict(torch.load(name_pth_E, weights_only=True, map_location='cpu'))

    loaded_model_S.eval()
    # loaded_model_E.eval()

    with torch.no_grad():
        data_output_S = loaded_model_S(data_input).numpy()
        # data_output_E = loaded_model_E(data_input).numpy()

    outputs_S = postprocess_large_B_S(data_output_S)

    return outputs_S


def nn_large_B_region_core(data_input, target_region_id):
    target_region_id = int(target_region_id)

    name_pth_S = 'large_B_S_core_top5_R' + str(target_region_id) + '_1.pth'
    name_pth_E = 'large_B_E_core_top5_R' + str(target_region_id) + '_1.pth'

    loaded_model_S = large_B_model.NN_S_core(target_region_id)
    loaded_model_S.load_state_dict(torch.load(name_pth_S, weights_only=True, map_location='cpu'))
    loaded_model_E = large_B_model.NN_E_core(target_region_id)
    loaded_model_E.load_state_dict(torch.load(name_pth_E, weights_only=True, map_location='cpu'))

    loaded_model_S.eval()
    loaded_model_E.eval()

    with torch.no_grad():
        data_output_S = loaded_model_S(data_input).numpy()
        data_output_E = loaded_model_E(data_input).numpy()

    outputs_S, outputs_E = postprocess_large_B(data_output_S, data_output_E)

    ans = np.zeros(8)

    id_argmax = np.argmax(outputs_S)

    ans[0] = int(num_region_row_core_large[int(target_region_id - 1)] + id_argmax)
    ans[1] = outputs_S[id_argmax]
    ans[2:8] = outputs_E[int(6 * id_argmax):int(6 * id_argmax + 6)] * 1000000

    return ans


def nn_large_B(data_FM):
    data_input = torch.tensor(preprocess_large_B(data_FM))

    data_region_S = np.zeros(4693)

    file_region_core_top5_elem = 'region_core_top5_elem_large.csv'
    data_region_core_top5_elem = pd.read_csv(file_region_core_top5_elem, header=None, encoding="utf-8").to_numpy()

    ans_IDSE = np.zeros(8)

    for i in range(8):
        outputs_S = nn_large_B_region(data_input, int(i + 1))

        data_region_S[int(num_region_row_large[i]):int(num_region_row_large[i] + num_each_region_large[i])] = outputs_S

        ans_i = nn_large_B_region_core(data_input, int(i + 1))

        if ans_i[1] > ans_IDSE[1]:
            ans_IDSE[0] = int(data_region_core_top5_elem[int(ans_i[0])][0] - 1)
            ans_IDSE[1:8] = ans_i[1:8]

    return data_region_S, ans_IDSE
