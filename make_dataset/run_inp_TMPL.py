import subprocess
import os

path_inp_folder = 'TBDPATHINPFOLDER'
job_id_start = TBDJOBIDSTART
job_id_end = TBDJOBIDEND

max_num_subprocess = 10
process_pool = []

jobid = job_id_start
while jobid <= job_id_end:
    if len(process_pool) < max_num_subprocess:
        file_inp = 'Job-' + str(jobid) + '.inp'
        path_inp = os.path.join(path_inp_folder, file_inp)
        if os.path.exists(path_inp):
            str_command = 'call abaqus job=Job-' + str(jobid) + ' cpus=32 int ask=off\n'
            pro = subprocess.Popen(str_command, shell=True)
            process_pool.append(pro)
            print('Job-{} submitted.'.format(str(jobid)))
        jobid += 1
    else:
        while all((p.poll() is None) for p in process_pool):
            pass
        for p in process_pool:
            if p.poll() is not None:
                process_pool.remove(p)
