from shutil import copyfile
import os
from tqdm import tqdm
from multiprocessing import Process


input_path = "/nxchinamobile2/shared/jjh/laion115m"
output_path = "/nxchinamobile2/shared/jjh/laion115m_grounding"


def copy_file(input_path, output_path):
    copyfile(input_path, output_path)


if __name__ == "__main__":
    process_list = []
    for dir_n in os.listdir(input_path):
        input_dir_path = os.path.join(input_path, dir_n)
        output_dir_path = os.path.join(output_path, dir_n)
        for file in tqdm(os.listdir(input_dir_path)):
            if file.endswith(".update"):
                p = Process(target=copy_file, args=(os.path.join(input_dir_path, file), os.path.join(output_dir_path, file[:-7])))
                p.start()
                process_list.append(p)
                if len(process_list) >= 256:
                    for p in process_list:
                        p.join()
                    process_list = []
                # copy_file(os.path.join(input_dir_path, file), os.path.join(output_dir_path, file[:-7]))

    # for dir_i, dir in enumerate(os.listdir(output_path)):
    #     print("processing dir {} {}/{}...".format(os.listdir(output_path), dir_i, len(os.listdir(output_path))))
    #     output_dir_path = os.path.join(output_path, dir)
    #     files = os.listdir(output_dir_path)
    #     tar_path = os.path.join(input_path, dir)
    #     for file in tqdm(files):
    #         tar_id = file.split('.')[0]

# for dir_n in os.listdir(input_path):

