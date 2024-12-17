import multiprocessing.process
import shutil
import subprocess
import numpy as np
import multiprocessing
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import inspect
import copy
from flightgrapher import flightgrapher
from plotlandingscatter import plotlandingscatter
from PlotLandingScatter.launch_site.launch_site_info import OtherSite
from PlotLandingScatter.launch_site.launch_site_info import LaunchSiteInfo

MODE_SINGLE = 'single'
MODE_MULTI  = 'multi'
MODE_AUTO   = 'auto'

def main():

    print("\n6DoF Rocket Simulator QUABLA by STEP... \n")
    print("-----------\n")

    # Pre Proc.
    job_list  = __pre_proc()
    # Main Proc.
    post_list = __main_proc(job_list)
    # Post Proc.
    __post_proc(job_list, post_list)

def __pre_proc():

    config_org, mode, nproc = __get_sim_condition()

    if mode == MODE_SINGLE:
        nproc = 1

    job_list = []
    if mode == MODE_SINGLE or mode == MODE_MULTI:
        job_dict = dict()
        job_dict['Config'] = config_org
        job_dict['Type']   = mode
        job_dict['nProc']  = nproc
        job_dict['Name']   = ''
        job_list.append(job_dict)
    
    elif mode == MODE_AUTO:

        path_job_case = './config/job_case.json'
        job_list = __make_job_list(path_job_case, config_org)

    return job_list

def __main_proc(job_list):

    post_proc_list = []
    for job in job_list:
        post_proc = main_task(job['Config'], job['Type'], job['nProc'], job['Name'])
        post_proc_list.append(post_proc)

    return post_proc_list

def __post_proc(job_list, post_proc_list):

    p_list = []
    for post, job in zip(post_proc_list, job_list):
        
        if job['Type'] == MODE_SINGLE:
            p = multiprocessing.Process(target=flightgrapher, args=post)
        if job['Type'] == MODE_MULTI:
            p = multiprocessing.Process(target=plotlandingscatter, args=post)
        p_list.append(p)
        p.start()

        for j, _p in enumerate(p_list):
            if not _p.is_alive():
                p_list.pop(j)

        # 使用プロセス数が上限に達したらプロセス終了を待つ
        if len(p_list) >= multiprocessing.cpu_count() - 1:
            # いずれかのプロセスの終了を待つ
            loopf=True
            while loopf:
                for j, _p in enumerate(p_list):
                    if not _p.is_alive():
                        p_list.pop(j)
                        loopf=False
                        break

    for p in p_list:
        p.join()

def main_task(config_org, type_task, nproc, name_task=''):
    '''
    Args:
        config      : 機体諸元のjsonファイル
        type_task   : ソルバーの計算種類。singleかmultiを指定
        nproc       : 並列数
        name_task   : 計算のタスク名（モデル名の下位名称）
    '''

    # print('STATUS: DEBUG', inspect.currentframe().f_code.co_name)

    # タスク名が存在する場合はモデル名の末尾につける
    config = config_org
    if name_task:
        config['Solver']['Name'] = config['Solver']['Name'] + '_' + name_task

    path_config = config['Solver']['Filepath']

    __print_quabla_AA()
    # jsonファイル読み込み
    root_path_result, model_name, launch_site, safety_exist, exist_payload = __read_json_config(config)
    # 結果ディレクトリ作成
    path_result = __make_result_dir(root_path_result, type_task, model_name)
    # ファイルバックアップ
    __copy_config_files(config, path_result)
    # 射点選択
    launch_site_info = __select_launch_site(config, launch_site)
    # 情報表示
    __print_calc_info(path_config, path_result, model_name, type_task, nproc, launch_site_info.site_name)
    # 計算実行
    subprocess.run(["java", "-jar", "Quabla.jar", path_config, type_task, path_result, str(nproc)], \
                    check=True)
    
    # Post Proc. 
    path_result += os.sep
    if type_task == MODE_SINGLE: 
        # return lambda: flightgrapher(path_result, launch_site_info, safety_exist, exist_payload)
        return (path_result, launch_site_info, safety_exist, exist_payload)

    elif type_task == MODE_MULTI:
        # return lambda: plotlandingscatter(path_result, str(config['Launch Condition']['Launch Elevation [deg]']), launch_site_info, exist_payload, 'y', safety_exist)
        return (path_result, str(config['Launch Condition']['Launch Elevation [deg]']), launch_site_info, exist_payload, 'y', safety_exist)
    

def __get_sim_condition():
    '''
    シミュレーションの計算モード，諸元の取得
    Return:
        config      : 機体諸元のjsonファイル
        mode        : ソルバーの計算種類。singleかmultiを指定
        nproc       : 並列数
    '''

    # print('STATUS: DEBUG', inspect.currentframe().f_code.co_name)

    # ファイル選択
    print('Rocket configuration files')
    path_config = (input("Enter the path of rocket parameter file (...json):\n >> ")).strip()
    while(not (os.path.exists(path_config) and path_config.endswith('.json')) ):
        
        # Debug mode
        if path_config == 'deb':
            path_config = 'config' + os.sep + 'sample_rocket.json'
            break

        path_config = (input('\nPlease enter again. \
                                 \nEnter the path of rocket parameter file (...json): \
                                 \n >> ')).strip()
        
    # configファイルのロード
    config = json.load(open(path_config, 'r', encoding="utf-8"))
    config['Solver']['Filepath'] = path_config
    
    # モード選択
    mode = input("\nEnter simulation mode (single or multi or auto):\n >> ")
    while(    mode != MODE_SINGLE 
          and mode != MODE_MULTI 
          and mode != MODE_AUTO
         ):
        mode = input("\nEnter simulation mode (single or multi):\n >> ")

    # 並列数の指定
    nproc = multiprocessing.cpu_count() - 1

    return config, mode, nproc

def pre_proc_gui(path_config, name_suffix=''):

    print('STATUS: DEBUG', inspect.currentframe().f_code.co_name)

    # jsonファイル読み込み
    config = json.load(open(path_config, 'r', encoding="utf-8"))
    
    # タスク名が存在する場合はモデル名の末尾につける
    config['Solver']['Name'] = config['Solver']['Name'] + name_suffix

###############################################################################
# Private Function(Sub-Routine)
###############################################################################

def __copy_config_files(json_config, path_result):

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size']   = 12
    plt.rcParams['figure.titlesize'] = 13
    plt.rcParams["xtick.direction"]   = "in"
    plt.rcParams["ytick.direction"]   = "in"
    plt.rcParams["xtick.top"]         = True
    plt.rcParams["ytick.right"]       = True
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["axes.linewidth"] = 1.5

    dir_config = path_result + os.sep + '_00_config'
    model_name = json_config['Solver']['Name']
    os.mkdir(dir_config)

    json_copy = json_config
    json_copy['Solver']["Result Filepath"] = dir_config

    shutil.copy(json_config["Engine"]["Thrust Curve"], dir_config + os.sep + model_name + '_thrust.csv')
    time_burn = json_config["Engine"]["Burn Time [sec]"]
    json_copy["Engine"]["Thrust Curve"] = dir_config + os.sep + model_name + '_thrust.csv'

    # Thrust
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot()
    ax.set_title('Thrust vs. Time')
    thrust_array = np.loadtxt(json_config["Engine"]["Thrust Curve"], delimiter=',', skiprows=1)
    ax.plot(thrust_array[:, 0], thrust_array[:, 1], color='#FF4B00')
    ax.axvline(x=time_burn, color='black', linestyle='--', linewidth=2)
    ax.set_xlim(xmin=0., xmax=thrust_array[-1, 0])
    ax.set_ylim(ymin=0.)
    ymin, ymax = ax.get_ylim()
    ax.text(x=time_burn, y=0.95*ymax, s=' Burning Time', horizontalalignment='right', verticalalignment='top', rotation=90)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Thrust [N]')
    ax.grid()
    fig.savefig(dir_config + os.sep + '_thrust.png', bbox_inches='tight', pad_inches=0.1)

    # Wind
    if json_copy["Wind"]["Wind File Exist"]:
        shutil.copy(json_config["Wind"]["Wind File"], dir_config + os.sep + model_name + '_wind.csv')
        json_copy["Wind"]["Wind File"] = dir_config + os.sep + model_name + '_wind.csv'

        wind_array = np.loadtxt(json_config["Wind"]["Wind File"], delimiter=',', skiprows=1)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].set_title('Wind Speed vs. Altitude')
        axes[0].plot(wind_array[:, 1], wind_array[:, 0], color='#FF4B00', marker='o')
        axes[0].set_ylim(ymin=0.)
        axes[0].set_xlabel('Wind Speed [m/s]')
        axes[0].set_ylabel('Altitude [m]')
        axes[0].grid()
        axes[1].set_title('Wind Direction vs. Altitude')
        axes[1].plot(wind_array[:, 2], wind_array[:, 0], color='#FF4B00', marker='o')
        axes[1].set_ylim(ymin=0.)
        axes[1].set_xlabel('Wind Direction [deg]')
        axes[1].set_ylabel('Altitude [m]')
        axes[1].grid()
        fig.subplots_adjust(wspace=0.4)
        fig.savefig(dir_config + os.sep + '_wind.png')

    else:
        json_copy["Wind"]["Wind File"] = ''

    # Cd
    if json_copy["Aero"]["Cd File Exist"]:
        shutil.copy(json_config["Aero"]["Cd File"], dir_config + os.sep + model_name + '_Cd.csv')
        json_copy["Aero"]["Cd File"] = dir_config + os.sep + model_name + '_Cd.csv'

        fig, ax = plt.subplots()
        ax.set_title('$C_D$ vs. Mach')
        Cd_array = np.loadtxt(json_config["Aero"]["Cd File"], delimiter=',', skiprows=1)
        ax.plot(Cd_array[:, 0], Cd_array[:, 1], color='#FF4B00', marker='o')
        ax.set_xlim(xmin=0.)
        ax.set_xlabel('Mach Number [-]')
        ax.set_ylabel('$C_D$ [-]')
        ax.grid()
        fig.savefig(dir_config + os.sep + '_Cd.png')

    else:
        json_copy["Aero"]["Cd File"] = ''

    # Lcp
    if json_copy["Aero"]["Length-C.P. File Exist"]:
        shutil.copy(json_config["Aero"]["Length-C.P. File"], dir_config + os.sep + model_name + '_Lcp.csv')
        json_copy["Aero"]["Length-C.P. File"] = dir_config + os.sep + model_name + '_Lcp.csv'

        fig, ax = plt.subplots()
        ax.set_title('$L_{C.P.}$ vs. Mach')
        lcp_array = np.loadtxt(json_config["Aero"]["Length-C.P. File"], delimiter=',', skiprows=1)
        ax.plot(lcp_array[:, 0], lcp_array[:, 1], color='#FF4B00', marker='o')
        ax.set_xlim(xmin=0.)
        ax.set_xlabel('Mach Number [-]')
        ax.set_ylabel('$L_{C.P.}$ [m]')
        ax.grid()
        fig.savefig(dir_config + os.sep + '_Lcp.png')

    else:
        json_copy["Aero"]["Length-C.P. File"] = ''

    # CNa
    if json_copy["Aero"]["CNa File Exist"]:
        shutil.copy(json_config["Aero"]["CNa File"], dir_config + os.sep + model_name + '_CNa.csv')
        json_copy["Aero"]["CNa File"] = dir_config + os.sep + model_name + '_CNa.csv'

        fig, ax = plt.subplots()
        ax.set_title('$C_{Na}$ vs. Mach')
        CNa_array = np.loadtxt(json_config["Aero"]["CNa File"], delimiter=',', skiprows=1)
        ax.plot(CNa_array[:, 0], CNa_array[:, 1], color='#FF4B00', marker='o')
        ax.set_xlim(xmin=0.)
        # ax.set_ylim(ymin=0.)
        ax.set_xlabel('Mach Number [-]')
        ax.set_ylabel('$C_{Na}$ [-]')
        ax.grid()
        fig.savefig(dir_config + os.sep + '_CNa.png')

    else:
        json_copy["Aero"]["CNa File"] = ''

    json.dump(json_copy, open(dir_config + os.sep + model_name + '_config.json', 'w', encoding="utf-8"), indent=4, ensure_ascii=False)

def __read_json_config(config):

    path_result   = config['Solver']['Result Filepath']
    model_name    = config['Solver']['Name']
    launch_site   = config['Launch Condition']['Site']
    safety_exist  = config['Launch Condition']['Safety Area Exist']
    exist_payload = config['Payload']['Payload Exist']

    return path_result, model_name, launch_site, safety_exist, exist_payload

def __make_result_dir(path, mode, model):
    '''
    Args:
        config     : 機体諸元のjsonファイル
        mode       : ソルバーの計算種類。singleかmultiを指定
        model_name : 
    '''

    # print('STATUS: DEBUG', inspect.currentframe().f_code.co_name)

    path_dir = path
    if not path_dir: 
        path_dir += '.' 

    # Make Result directory
    result_dir = __make_directory(path_dir + os.sep + 'Result_' + mode + '_' + model)

    return result_dir

def __select_launch_site(config, launch_site):

    if launch_site == '0' :
        launch_site_info = OtherSite()
        launch_site_info.launch_LLH[0] = float(config['Launch Condition']['Launch lat'])
        launch_site_info.launch_LLH[1] = float(config['Launch Condition']['Launch lon'])
        launch_site_info.launch_LLH[2] = float(config['Launch Condition']['Launch height'])
        launch_site_info.center_circle_LLH = launch_site_info.launch_LLH

    else:
        launch_site_info = LaunchSiteInfo(launch_site)

    return launch_site_info

def __make_job_list(path: str, config_org: json):
    '''
    Args:
        path        : ジョブリストのパス
        config_org  : 複製元となるマスター諸元のjson
    '''

    path_dir    = config_org['Solver']['Result Filepath']
    model       = config_org['Solver']['Name']
    path_result = __make_result_dir(path_dir, MODE_AUTO, model)
    # マスター諸元の結果出力用パスの変更
    config_org['Solver']['Result Filepath'] = path_result

    json_job = json.load(open(path, 'r', encoding="utf-8"))
    json_job_list = json_job.get('Job')
    job_list = []

    dir_configs = config_org['Solver']['Result Filepath'] + os.sep + 'config'
    os.mkdir(dir_configs)

    for job in json_job_list:

        mode = job.get('Mode')
        config = copy.deepcopy(config_org)
        name_config = config_org['Solver']['Name']
        name_job = ''
        job_dict = {}
        job_dict['Type'] = mode
        
        if mode == MODE_SINGLE:
            name_job = job.get('Name')
            job_dict['Name']  = job.get('Name')
            job_dict['nProc'] = 1
            model_wind = job.get('Wind Model')

            if model_wind == 'No':
                config['Wind']['Wind File Exist']  = False
                config['Wind']['Wind File']        = ''
                config['Wind']['Wind Model']       = 'law'
                config['Wind']['Wind Speed [m/s]'] = 0.
            
            elif model_wind == 'Power':
                config['Wind']['Wind File Exist'] = False
                config['Wind']['Wind File']       = ''
                config['Wind']['Wind Model']      = 'law'
            
            elif model_wind == 'Original':
                config['Wind']['Wind File Exist'] = True

        elif mode == MODE_MULTI:
            elevation  = config['Launch Condition']['Launch Elevation [deg]'] + job.get('Elevation Error [deg]')
            name_job = str(elevation) + 'deg'
            job_dict['Name']  = str(elevation) + 'deg'
            job_dict['nProc'] = min([job.get('nProc'), multiprocessing.cpu_count() - 1])
            config['Launch Condition']['Launch Elevation [deg]'] = elevation

        name_config += '_' + name_job + '_config.json'
        path_config = dir_configs + os.sep + name_config
        config['Solver']['Filepath'] = path_config
        json.dump(config, open(path_config, 'w', encoding="utf-8"), indent=4, ensure_ascii=False)
        job_dict['Config'] = config
        job_list.append(job_dict)

    return  job_list

def __make_directory(result_dir):
        
    _dir_result = result_dir
    if os.path.exists(_dir_result):
        resultdir_org = _dir_result
        i = 1
        while os.path.exists(_dir_result):
            _dir_result = resultdir_org + '_%02d' % (i)
            i += 1
    os.mkdir(_dir_result)

    return _dir_result

def __print_calc_info(path_config, path_result, model, mode, nproc, site):

    print('-------------------- INFORMATION --------------------')
    print('  Config File:     ', os.path.basename(path_config))
    print('  Model Name:      ', model)
    print('  Simulation Mode: ', mode)
    print('  CPU Count:       ', str(nproc))
    print('  Result File:     ', os.path.basename(path_result))
    print('  Launch Site:     ', site)
    print('-----------------------------------------------------\n')

def __print_quabla_AA():
    '''
    Quablaのアスキーアート表示
    '''

    print('''\

    　 ／＼
    　｜Q  ∧∧
    　｜U ( ﾟДﾟ)
    　｜A ⊂STEP|
    　｜B ⊂__ ノ＠
    　｜L ｜
    ／｜A ｜＼
    ￣￣￣￣￣
    　　ε
    　　ε
    ''')
    print('Quabla Start...\n')

def debug_cui():

    config, mode, nproc = __get_sim_condition()
    main_task(config, mode, nproc)

def debug_auto():

    # Pre Proc.
    config_org, mode, nproc = __get_sim_condition()

    path_dir = config_org['Solver']['Result Filepath']
    if not path_dir: 
        path_dir += '.' 
    model = config_org['Solver']['Name']
    path_result = __make_directory(path_dir + os.sep + 'Result_' + 'auto_' + model)
    config_org['Solver']['Result Filepath'] = path_result

    job_list = __make_job_list('/Users/kenta/Downloads/job_case.json', config_org)

    # Main Proc.
    post_proc_list = []
    for job in job_list:
        post_proc = main_task(job['Config'], job['Type'], job['nProc'], job['Name'])
        post_proc_list.append(post_proc)

    # Post Proc.
    p_list = []
    for post, job in zip(post_proc_list, job_list):
        
        if job['Type'] == MODE_SINGLE:
            p = multiprocessing.Process(target=flightgrapher, args=post)
        if job['Type'] == MODE_MULTI:
            p = multiprocessing.Process(target=plotlandingscatter, args=post)
        p_list.append(p)
        p.start()
        # idx += 1

        for j, _p in enumerate(p_list):
            if not _p.is_alive():
                p_list.pop(j)

        # 使用プロセス数が上限に達したらプロセス終了を待つ
        if len(p_list) >= multiprocessing.cpu_count() - 1:
            # いずれかのプロセスの終了を待つ
            loopf=True
            while loopf:
                for j, _p in enumerate(p_list):
                    if not _p.is_alive():
                        p_list.pop(j)
                        loopf=False
                        break

    for p in p_list:
        p.join()

if __name__=='__main__':

    main()
    # debug_cui()
    # debug_auto()
