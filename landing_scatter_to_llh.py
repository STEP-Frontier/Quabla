"""落下分散CSV(射点原点の相対NED[m])を緯度経度に変換するスクリプト.

Quabla本体には手を加えず, MultiSolverが出力した
`trajectory<elev>[deg].csv` / `parachute<elev>[deg].csv` / `payload<elev>[deg].csv`
を読み込んで, 各落下点の緯度・経度をフラットな表形式のCSVに書き出す.

Usage:
    python landing_scatter_to_llh.py <config.json> <landing.csv> [<landing.csv> ...]

    # 複数まとめて (Windowsのcmd/PowerShellではワイルドカードも可)
    python landing_scatter_to_llh.py config/Volubilis_83.0deg_config.json ^
        "Result_multi_Volubilis_83.0deg/*.csv"

変換後のファイルは入力CSVと同じディレクトリに `<元の名前>_LLH.csv` として保存される.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import PlotLandingScatter.coordinate as cd


# MultiSolverが落下地点として保存しているのは射点を原点としたNED[m]で,
# CSVでは風速ごとに "x" 行 = North, "y" 行 = East の2行1組で並んでいる.
# (src/quabla/simulator/MultiSolver.java: getWindMap, src/quabla/output/OutputLandingScatter.java)
# 磁気偏角はVariableTrajectoryの初期姿勢生成時点で補正済みなので, ここでは真北基準として扱う.


def load_json(filepath):
    """configのJSONを読み込む. UTF-8でもShift-JIS(cp932)でも読めるようにする."""
    for encoding in ('utf-8-sig', 'utf-8', 'cp932'):
        try:
            with open(filepath, mode='r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError('config JSONを読み込めませんでした: ' + filepath)


def get_launch_llh(config, args):
    """射点の[緯度, 経度, 高度]を取得する.

    Rocket.java(252-254行)と同じくconfigの"Launch lat"/"Launch lon"/"Launch height"を使う.
    コマンドラインで--lat/--lon/--heightが指定されていればそちらを優先する.
    """
    launch_cond = config['Launch Condition']
    lat = args.lat if args.lat is not None else float(launch_cond['Launch lat'])
    lon = args.lon if args.lon is not None else float(launch_cond['Launch lon'])
    height = args.height if args.height is not None else float(launch_cond['Launch height'])

    return np.array([lat, lon, height])


def get_base_azimuth(config):
    """風向の基準角[deg]を取得する.

    OutputLandingScatter.javaのヘッダは 0, step, ..., 360 と基準角を含まない値で書かれているため,
    実際の風向を得るにはconfigの"Base Wind Azimuth [deg]"を足す必要がある.
    (MultiSolver.java: azimuthArray[i] = 360.0 * i / numAzimuth + azimuthBase)
    """
    return float(config['Multi Solver']['Base Wind Azimuth [deg]'])


def read_landing_scatter(filepath):
    """落下分散CSVを読み込んで (風速ラベル配列, 風向[deg]配列, NED配列) を返す.

    戻り値のNED配列は shape = (風速数, 風向数, 3) で, [North, East, 0.0].
    先頭と重複する末尾の風向(360deg)の列は除く.
    """
    with open(filepath, mode='r', encoding='utf-8') as f:
        rows = [row for row in csv.reader(f)]

    # 末尾の空セルを落とす
    rows = [[cell for cell in row if cell.strip() != ''] for row in rows]
    rows = [row for row in rows if len(row) > 0]

    azimuth_offset_array = np.array([float(v) for v in rows[0]])
    body = rows[1:]
    if len(body) % 2 != 0:
        raise ValueError('風速ごとにx行/y行が揃っていません: ' + filepath)

    # 末尾の360degは先頭の0degと同じ点なので除く
    num_azimuth = len(azimuth_offset_array)
    if num_azimuth > 1 and np.isclose(azimuth_offset_array[-1] - azimuth_offset_array[0], 360.0):
        num_azimuth -= 1
        azimuth_offset_array = azimuth_offset_array[:num_azimuth]

    num_speed = len(body) // 2
    speed_label_array = []
    pos_NED_array = np.zeros((num_speed, num_azimuth, 3))

    for i in range(num_speed):
        row_north = body[2 * i]
        row_east = body[2 * i + 1]
        speed_label_array.append(row_north[0])
        # 先頭2列は風速ラベルとx/yラベル
        pos_NED_array[i, :, 0] = np.array([float(v) for v in row_north[2:]])[:num_azimuth]
        pos_NED_array[i, :, 1] = np.array([float(v) for v in row_east[2:]])[:num_azimuth]

    return speed_label_array, azimuth_offset_array, pos_NED_array


def parse_speed(speed_label):
    """"1.0 m/s" のようなラベルから数値を取り出す."""
    return float(speed_label.replace('m/s', '').strip())


def convert(filepath, launch_LLH, azimuth_base, suffix):
    """1つの落下分散CSVを緯度経度のフラットなCSVに変換して保存する."""
    speed_label_array, azimuth_offset_array, pos_NED_array = read_landing_scatter(filepath)

    dirname = os.path.dirname(os.path.abspath(filepath))
    basename = os.path.splitext(os.path.basename(filepath))[0]
    filepath_out = os.path.join(dirname, basename + suffix + '.csv')

    with open(filepath_out, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Wind Speed [m/s]', 'Wind Azimuth [deg]', 'Latitude', 'Longitude'])

        # 風向は0deg始まりの昇順に並べ替える
        azimuth_array = (azimuth_offset_array + azimuth_base) % 360.0
        order = np.argsort(azimuth_array, kind='stable')

        for i, speed_label in enumerate(speed_label_array):
            speed = parse_speed(speed_label)
            for j in order:
                azimuth = azimuth_array[j]
                # ENU2LLHforKmlは[経度, 緯度, 0.0]を返す
                lon, lat, _ = cd.ENU2LLHforKml(launch_LLH, pos_NED_array[i, j])
                writer.writerow([
                    '{:g}'.format(speed),
                    '{:g}'.format(azimuth),
                    '{:.12f}'.format(lat),
                    '{:.12f}'.format(lon)])

    return filepath_out, len(speed_label_array) * len(azimuth_offset_array)


def expand_filepath(filepath_array):
    """ワイルドカードを展開しつつ, 変換対象のCSVを列挙する."""
    filepath_expanded = []
    for filepath in filepath_array:
        # ファイル名に "[deg]" のような角括弧が含まれるので, 実在するパスはそのまま使う
        if os.path.isfile(filepath):
            filepath_expanded.append(filepath)
            continue

        matched = [path for path in glob.glob(filepath) if os.path.isfile(path)]
        if len(matched) == 0:
            print('[Warning] ファイルが見つかりません: ' + filepath)
            continue
        filepath_expanded.extend(sorted(matched))

    return filepath_expanded


def main():
    parser = argparse.ArgumentParser(
        description='落下分散CSV(射点原点の相対NED)を緯度経度に変換する')
    parser.add_argument('config', help='シミュレーションに使用したconfigのJSONファイル')
    parser.add_argument('csv', nargs='+', help='変換する落下分散CSV (複数指定可)')
    parser.add_argument('--suffix', default='_LLH',
                        help='出力ファイル名に付ける接尾辞 (default: _LLH)')
    parser.add_argument('--lat', type=float, default=None, help='射点緯度[deg]の上書き')
    parser.add_argument('--lon', type=float, default=None, help='射点経度[deg]の上書き')
    parser.add_argument('--height', type=float, default=None, help='射点高度[m]の上書き')
    args = parser.parse_args()

    config = load_json(args.config)
    launch_LLH = get_launch_llh(config, args)
    azimuth_base = get_base_azimuth(config)

    print('Launch Point : lat = {:.9f} [deg], lon = {:.9f} [deg], height = {:.1f} [m]'.format(
        launch_LLH[0], launch_LLH[1], launch_LLH[2]))
    print('Base Wind Azimuth : {:g} [deg]'.format(azimuth_base))

    filepath_array = expand_filepath(args.csv)
    if len(filepath_array) == 0:
        print('[Error] 変換対象のCSVがありません')
        sys.exit(1)

    for filepath in filepath_array:
        try:
            filepath_out, num_point = convert(filepath, launch_LLH, azimuth_base, args.suffix)
        except (ValueError, IndexError) as e:
            print('[Skip] ' + filepath + ' : ' + str(e))
            continue
        print('[Done] ' + filepath + ' -> ' + filepath_out + ' ({} points)'.format(num_point))


if __name__ == '__main__':
    main()
