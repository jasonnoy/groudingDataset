import argparse
import os
import json
from tqdm import tqdm
from multiprocessing import Process
import webdataset
import math


offset_list = ['f1c7333cf2852dd27b9738c4fd6ea3ed', 'e9b2300cfb28f250510c7058621b8018', 'b66d7d86f4dd8f7a9307d73d5d5499b6', '083b016f57c600209fe7c2505e49dd03', '5a49ea713843e127a6f740b533ae4e96', '2639c886a9dd379e14bf2675e3bcf2fa', '3ee08c80b53333b0f2f5b89eb4a1aa40', '5709b064d6b1a942ad737062c11efac7', '12d38c4315353097c7b25b19a97bfce0', '28e65ae40c6dca21e417bf240b7c8904', '75a7e75158af463af195b80f468f38ec', '51798b37550813478ef08105faf356a2', '87449a758e2db158961c8e2122d876cf', 'c992011b65328ccfc6c59b444313deca', 'f174ade64310348c54d83a59d884a0fe', '09ea55dbc321b9520bf6d58a248da300', '1ea1b894e176bc0c1ed714ee7b5a8892', '7486cc4dcac70fd8a0d942be720e19c1', '9d2adb761d52cd080c60c0327b93d556', '70c34609270b8e7d06c953104e1a28a4', '97b92f794929a2d2213f39648fae4e23', 'd3a6b83e9a22f307d1dc1257c25d8cab', 'de857a0bd386623b0ae13c251ab8a1a3', '2ae38ee01898121917c0a9bcaf750179', '255065764c3f6ecb364179ef220eaa72', '4e7ad2f99c1dfed100a03bee8381bc7e', 'affa90f542753e56b5e4dfea3fbdfd9e', '0a6f18182c6be5a4ea6c09ddc4ab911b', '60d7f9a269df8e7c81317cdff4bcd435', 'ffeb991e30f98549a3003009bd5edb36', 'fc13547e6879d33f3df9a2ecea4c3c4e', '6c6b74dafd6c8c335c0b5f2f05865730', 'ebe3d20d2c56d26928d92fb48791054a', 'e4492dc944e15ae48fad76061ad9a486', 'aa439b8ffdcaed2ae775d965c0b1d4fb', '0366bb057bfbf15e4c4380a2b00eeec2', 'fc147876e71abdbbac323b6c0440ce72', '28abfa991d4d1a28f070140814dd4d73', 'eea5926788f0814a3ffe6032d45849d8', '989936c5c6268e2a5e1d7e8ed3aabc0e', '48b1e6d3051e79d5090ce46410289ef4', '691608ab7b5e80c34d417b97decc93b0', '1f90a5eb208109649b5a9885245162f0', '9d42f5de81f183a3cec926a3217aad21', '792c4a5643eefb90ec28ed660e75b701', '251f4b1446028aead7008099006f3e0c', 'e347087659a5ae52a876281142c17de6', '38696b44db32c74c68671ebd6590d64d', '2a9efb388f7b9078a8de36c04bd34506', '76b0984252004bd8353885767fc26f27', '1906d0c67ff84b28844f4c558fae3015', '30275c47480f5867644a743cd3d32c31', '45ad7161d7e7028ef4402fc3f8619266', '81a268ac57b704e81a51fe537a452b44', '5e8d7f4bea68a5015334a4b99bc94737', '62b2f9108d1abb5ce423a89b1b69cd63', '8a049b938b121236bef17e26e5bbd7a5', '90a6b283b7359aa987d3a187367bc324', '065297de12478466f6b191f207e02593', '34c8ed947072c7485139079f61adbd44', 'acc80402b0b7b468c2711fc1969cbace', '22153526f162c7e9020148a2ebe79496', '76b9956b17e4352e2ffd67c211e2a742', 'a8abd1a222d10c19f3a7cb72fecc8c6f', '6a72bcdac1b00ab420b711c8a41e3f8f', '15f37a3399536e7326a334ded62d6097', '3624c6e829b3f8b13ac664e0129e3386', '2a0270bae2185d26aefe5489dcbe19e6', '8f0a569c48d8144427064a92be621283', '82f5b334dbcdae4d46160b70d8690de2', 'f27fb6976b1a6e054e9ee7cacfa68c23', '2f19a85f400c4154c5856cad32136566', 'b265da7ac1999266f2ca2bd57899809e', 'c83914ff4f56d1903bab67376d9919e7', '63db12f078274262d18db038589a17f1', '0ff976f56f82d2683efe435bc73038f7', 'b7d534da28b4a419cbd8fc0a4329120b', '5bda13e08b342a78bb8287e9cfed5faa', 'fd396a8a7acb056f6043c308fe2e871a', 'b9ad5652183417b1e3302e494a0cedda', 'a74f35af79bcafa29819fce6b9b3e147', '37f25e2593c68aed912003fff738ddba', '4fa57db183f6233e2d8d51280f44df4c', '3108b6e68cc17e08d15fa70cca6067ce', '3507eae1e57c1b36bc5a4c175ac7f326', '2214ccf2a8431701ea8c4a5d342d1160', '3c11fa800ef82caf0ad45b58f3563d36', '138a4d3fb3e9cd6b15c208ecc8be1c65', 'e5e30f79be5d7f1791710cd6d632e030', '49d783bbbea7a0abb0ab5e6f7a52c650', 'd3b3792eba3046ffe364686e216fec6d', '9afb3a4fe25e0f037433205300d94712', '264be371242c925d876ddf558c5b5773', '7cdc2b4ef31d0d400ee3f54eac37325c', 'ac430dfe63ffbcf9d8784a2f4bb73183', '8ec771d47a7196e5583622f3d3568ee6', 'b5d9ea554abc7f881c4788a0fe00f8e9', 'c044af4e10673f32dfb1dd7d9999b3d2', 'fda7c4e9c1db346e7010cd03403e68c4', '969eb7c437e0af71373e26b402e81874', 'a2ef90b7f58a8c0f4d554dc131598220', '1ff8df7dd2aecb82c177ded72f18f3ac', 'b313585ba276231783eb0a6bc6a08ffc', '78d7b6476274f49183997739cc29bc81', '7c4aaa4a9621d1b6b262af6c02f1e506', '1ce1bb94f03379a711f3ee972ddb091b', 'acfbd90d8f2169ac9ebcb9a26aed0701', 'b66a015b873c074bc7ad182ae6b87760', '56af03cfefb4a5508d9ad41801ef2e71', '17b2e133ce4485ca070427726a769f43', '80ccc13ca721b2b5db4bb96b801b259c', '866cbf21a81f46f1faffd82fb12f40f0', 'a199b9494573223f51ff4d55dd31341e', 'ac96dd090b48379ef7c118b8a1544507', 'bdcb501d35e997f00a8a9a8b486bdeed', '713b77681fbd38d6b0b28da4463c7cbf', 'dc0bd8a6436fd5805fc0eb5427285b96', 'a0d02db239619d4dbf92800ba50d24f5', '7508092b1c58e846c4417d6d32e1f0f8', 'f15ecec2aeefde1d855306557cb28c41', '092d49f8d4502fa2767010144cbc2f2e', '45c943870bd0131c5f8b2b9f623a1055', '59aaa3526a3857a07e6814464afd0e07', '63cafcc2f33d0c403336c15a42caff4e', 'ebc85826fda80de66534f8bcabaa1ebf', '9a3828813559175ed13aedc94122d127', '1839eb241294d36dd5635101de90c581', 'ee021a8255a463d20288c922751de3d7', '51b6fbbfec6ef97663873c1fea931950', '5ba93c1493a7bcb9a7ef38383bb2a7c7', '273095c5eedc89b689361515605fbab8', '9b40d7a415cd3340817fd9eebb6a7c68', '9e4672ce62910aa8c896c5ae7e3e323c', '9fce6bee65cab4867c4f778037e63191', 'd31322ebca29e7280a9cd5dc0810fff1', '26ccb480c263317ef6c759343cdbbc88']

offset_ids = offset_list[1:]
offset_keys = offset_list[:-1]
offset_map = dict(zip(offset_keys, offset_ids))


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def combine_files(in_path, meta_path, correct_path, out_path, backup_path, record_path):
    data = webdataset.WebDataset(in_path)
    backup_dict = {}
    with open(backup_path, 'r', encoding='utf-8') as f:
        for line in f:
            back_data = json.loads(line)
            backup_dict[back_data['SAMPLE_ID']] = back_data
    with open(correct_path, 'r', encoding='utf-8') as f1, open(meta_path, 'r', encoding='utf-8') as f2, open(out_path, 'w', encoding='utf-8') as f3, open(record_path, 'a', encoding='utf-8') as f4:
        res_dict = {}
        for f1_line in f1:
            correct_data = json.loads(f1_line)
            res_dict[correct_data['SAMPLE_ID']] = correct_data
        f1.close()
        for f2_line in f2:
            meta_data = json.loads(f2_line)
            res_dict[correct_data['SAMPLE_ID']] = meta_data
        f2.close()
        for d in data:
            idx = d['id'].decode()
            # if idx in offset_map:
            #     idx = offset_map[idx]
            #     if idx in res_dict:
            if idx in res_dict:
                f3.write(json.dumps(res_dict[idx]) + '\n')
            else:
                f3.write(json.dumps(backup_dict[idx]) + '\n')
                if backup_dict[idx]['status'] == 'success':
                    f4.write(idx+'\n')
        f3.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
    debug_path = "/nxchinamobile2/shared/jjh/laion115m-debug"
    output_path = "/nxchinamobile2/shared/jjh/laion115m_grounding"
    ids = []
    for dir in os.listdir(debug_path):
        ids.extend(os.listdir(os.path.join(input_path, dir)))
    ids = [name.split(".")[0] for name in ids if name.endswith(".tar")]
    ids.sort()
    select_ids = split_list_by_n(ids, args.world_size)[args.rank]
    process_list = []
    for idx in tqdm(select_ids):
        dir_name = "part-000"+idx[:2]
        output_dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        meta_dir_path = os.path.join(debug_path, dir_name)
        tar_dir_path = os.path.join(input_path, dir_name)

        tar_filename = idx+".tar"
        tar_file_path = os.path.join(tar_dir_path, tar_filename)
        meta_filename = idx+".meta.jsonl"
        meta_file_path = os.path.join(meta_dir_path, meta_filename)
        backup_file_path = os.path.join(tar_dir_path, meta_filename)
        out_file_path = os.path.join(output_dir_path, meta_filename)
        correct_file_path = os.path.join(meta_dir_path, "corrected_"+meta_filename)
        record_path = os.path.join(output_path, "record.txt")
        p = Process(target=combine_files, args=(tar_file_path, meta_file_path, correct_file_path, out_file_path, backup_file_path, record_path))
        p.start()
        process_list.append(p)
        if len(process_list) >= 36:
            for p in process_list:
                p.join()
            process_list = []
