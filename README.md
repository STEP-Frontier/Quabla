Quabla
==
6-DoF Rocket Simulator for STEP<br>
coding UTF-8<br>
released under [MIT License](LICENSE)

<img src="./Quabla_logo.png" width="400px">

# Features
* 高速な軌道計算ソルバー
   * Javaによる高速な計算実行
   * メモリ使用量などに配慮したチューニングの実施
   * 時間刻み自動調整機能（適応時間刻み，Adaptive Time Stepping）付き微分方程式ソルバーを用いた計算ステップの省略
   * Javaのマルチスレッドを用いた落下分散計算次の並列処理
* Pythonを用いた豊富なグラフ描画
   * 審査書作成に必要な図のほとんどを網羅
   * グラフの見やすさに配慮した配色
   * 飛翔モードごとに線種の使い分け
   * 落下分散や飛翔軌道などのkmlファイル出力
* 容易な環境構築
   * 導入が比較的簡単なJavaとPythonのインストールのみ
   * Java関係のライブラリはセットアッププログラムによる自動インストール
   * Python関係のライブラリはAnacondaを併用することで大幅短縮可能
* その他便利機能の搭載
   * 機体諸元の簡易サマリー(rocket_param.txt)，計算結果サマリー(result.txt)の出力
   * 風向風速制限自動判定および制限表出力(judge.csv)

# Flight Event
以下の飛行イベントに対応している。
* ノミナルフライト
* ランチャ上での挙動
* 予報風，統計風を用いた飛翔シミュレーション（csvによる風データの入力）
* パラシュートによる減速落下
   * タイマー指令，頂点検知可能
   * 開傘ラグの考慮（頂点到達からパラシュート放出までの遅れ時間）
   * 二段分離（高度検知，タイマー指令可能）
* 超音速パラメータのマッハ数依存性考慮
   * 抗力（軸力）係数，法線力係数傾斜，圧力中心のマッハ数変化のcsvファイルの読み込み
* ペイロード分離

# Reference
* 戸川隼人，石黒登美子；スピンを伴うロケットの運動を計算するプログラム，航空宇宙技術研究所資料 NAL TM-145，1968
* 嶋田有三，佐々修一；飛行力学，森北出版，2017

# 開発環境
* [Eclipse](https://mergedoc.osdn.jp/) <br>
* VsCode (2024/05現在)

# Requirements
## Java
  以下のサイトから最新のjdkをインストール  
  https://www.oracle.com/java/technologies/javase-jdk16-downloads.html  
  その後，以下のサイトを参考に環境変数を通す(Windowsのみ)  
  https://www.javadrive.jp/start/install/index4.html  

  以下のコマンドでJavaのバージョンが表示されることを確認する。
  ```
  $ java -version
  java version "17.0.1" 2021-10-19 LTS
  Java(TM) SE Runtime Environment (build 17.0.1+12-LTS-39)
  Java HotSpot(TM) 64-Bit Server VM (build 17.0.1+12-LTS-39, mixed mode, sharing)
  ```
  また，コンパイルに`javac`と`jar`を使用するのでこちらもパスが通っているかを確認する。
  ```
  $ javac -version
  javac 17.0.1
  $ jar
  使用方法: jar [OPTION...] [ [--release VERSION] [-C dir] files] ...
  詳細は、`jar --help'を実行してください。
  ```
   (Macはターミナルで`java --version`)  

## Python
  PythonもJavaと同様にインストールを行い，パスを通す。
  以下のコマンドでパスが通ってるか確認する。
  ```
  $ python -V
  Python 3.9.7
  ```

Pythonを使用する場合，あらかじめライブラリがそろっているAnacondaが便利である。
Anacondaを用いる場合，使用法は以下の通り。

### anacondaの使用方法
  以下のサイトからダウンロード。  
  https://www.anaconda.com/products/individual  

  さらにanacondaに含まれないライブラリとしてsimplekmlおよびPolycirclesを使用するのでそれもダウンロードする。  
  この際にWindowsとMacでは少々方法が異なる。

  * Windows  
  anaconda promptを起動。  
  `conda install -c conda-forge simplekml`と入力。  
  `conda install -c conda-forge polycircles`と入力。  

  * Mac  
  ターミナルを起動。  
  `conda activate`と入力。  
  `conda install -c conda-forge simplekml`と入力。  
  `conda install -c conda-forge polycircles`と入力。

  なお，Macにおいて終了後に以下のコマンドを入力することでanacondaを終了する。  
  `conda deactivate`  
  `conda config --set auto_activate_base False`

# Libraries
## Java
Jsonファイルを読み込むために，以下のライブラリが必要。
Jacksonで読み込んでいる。
* jacson-core
* jackson-annotations
* jackson-databind
以上のJava用のライブラリは`setup_jackson.py`実行時にダウンロードされるため，
後述のInstallationのコマンドをすべて実行している場合はインストール不要である．

# Installation(Compile)
gitからcloneしたらQuablaのフォルダに移動する。
```
$ cd lib
$ python setup_jackson.py
$ cd ..
$ python setup.py
```

# 【読まなくていいです】Useage(Eclipseを用いて実行可能jarを作る場合)
Eclipseを用いて実行可能jarを作る場合，以下の手順に従う。
開発者以外は読み飛ばして問題ない。
1. 最新のEclipseをインストールする。JavaのFull Editionでよい。（すでにEclipseをインストールしてる場合は飛ばす。）
2. 本レポジトリをクローン&インポートする。
[こちらのサイト](https://rainbow-engine.com/github-eclipse-connect/)を参考にするとよい。クローンするとき，「クローン終了後，すべての既存Eclipseプロジェクトをインポート」にチェックを入れることを忘れない。また，クローン後にプロジェクトをインポートしないとJavaのパースペクティブに表示されない。
3. `src/quabla/QUABLA.java`を右クリックし，`実行>実行の構成`を選ぶ。Javaアプリケーションを選択し，引数のタブから`プログラムの引数`に`config\sample_rocket.json single`を入力して実行する。
4. Javaプロジェクト上で（Quablaフォルダを）右クリックし，`エクスポート`を選択し，`Java>実行可能JARファイル`を選択し，`次へ`を選択。起動構成は先ほど実行した，`QUABLA - Quabla`を選択し，エクスポート先は`Quabla.py`と同じ階層にする。`完了`を押すとコンパイルが行われる。警告がいっぱい出るけど気にしない。
5. `Quabla.jar`が指定した場所に生成されていればコンパイル成功。
6. あとはanacondaとかで適当に`Quabla.py`を実行すればいい。
Eclipseを使用して編集したい場合はPyDevとPythonをインストールする。
<!-- 1. 実行時のコマンドラインで機体の諸元を入力したjsonファイルを指定。<br>
パスの指定方法は相対パスでも絶対パスでもどちらでもよい。
例えば，あらかじめ入っている`sample_rocket.json`を相対パスで指定する場合，
コマンドライン引数は`sample_rocket.json`となる。<br>
`C:\hoge`にある`rocket_config.json`を絶対パスで指定する場合，
コマンドライン引数は`C:\hoge\rocket_config.json`となる。
2. `QUABLA.java`を実行。 -->

# Execute

0. anaconda promptを起動し，`Quabla.py`がある階層まで移動する(Windows)。
Macの場合は，ターミナルで`conda activate`と入力。

1. `Quabla.py`を実行する。
```
$ python Quabla.py
```

2. 以下のようにRocket configファイルを聞かれるので，計算したい機体のconfigファイルを指定する。
```
Rocket configuration files
Enter the path of rocket paramater file (...json):
 >>
```
<!-- `config/`フォルダ内の`.json`ファイルのみ一覧に表示される。
機体のconfigファイルは`config`フォルダ内に格納する。
例えば，上の`sample_rocket.json`を指定する場合，`sample_rocket.json`と入力して，Enterキーを押す。 -->
計算したい機体データの`**.json`ファイルをanaconda prompt（Macはterminal）にドラッグ&ドロップし，Enterキーを押す．**（ドラッグ&ドロップ推奨）**
~~（Macの場合，ファイル末尾に半角スペースが追加されてしまうため，末尾の半角スペースを消す．）~~<br>
`**.json`ファイルの絶対パスまたは相対パスを直接入力しても問題ない．
ここで相対パスを用いた場合，jsonファイル内でパスを使用しているファイルのパスも相対パスで指定する必要がある。
絶対パスを用いた場合は、jsonファイル内でも推力履歴等のファイルの場所を絶対パスで指定する必要がある。
絶対パスで統一することを強く推奨する。
`**.json`ファイルの各パラメータの説明は後述．
サンプルの機体データとして，`config/`フォルダ内に`sample_rocket.json`が格納されている．

3. 次に，以下のようにシミュレーションモードを聞かれるので指定する。
```
Enter simulation mode (single or multi):
 >>
```
`single`か`multi`を指定する。
`single`,  `multi`の各モードについては以下の通り。<br>
|項目|説明|
|---|---|
|`single`|単一条件での計算。位置や姿勢角の時間履歴などを見たい場合はこのモードを選択。|
|`multi`|複数条件での計算。落下分散を計算したい場合に選択。singleモードと異なり位置などの時間履歴は出力されず，落下地点や最高高度などの表のみ出力。|

# Input
## Rocket Configurations
### Caution
* jsonの文法に従って記入すること。
例えば，
	<!-- * 負の値を入力している（特に減衰モーメント係数。プログラム内で自動で修正してくれる） -->
	* コロンが無い，逆にコロンが必要ない <br>
などの文法ミスに注意。

* jsonファイルをUTF-8で編集しているか必ず確認

* 長さなどの定義に注意。長さの基準が異なっている可能性がある。

### Solver
ソルバーに関する設定。
|項目|備考|
|---|---|
|Name|プロジェクト名や機体名など|
|Result Filepath|結果の出力先。|
|Time Step|シミュレーションの時間刻み。|

### Multi Solver
複数条件のシミュレーション時の設定。
|項目|単位|備考|
|---|---|---|
|Minimum Wind Speed | m/s | 計算する風速の最小値。|
|Step Wind Speed | m/s || 風速の時間刻み。|
|Number of Wind Speed | - | 計算する風速の数。|
|Number of Wind Azimuth | -  | 計算する風向の数。基本的に4の倍数にすること。|
|Base Wind Azimuth [deg]| -  | 風向の基準。例えば，打上方位角と同じ場合，向かい風基準で，風向が適用される。 |

### Launch
ランチャなどの打上げ条件に関する設定。<br>
|項目|単位|備考|
|---|---|---|
|Date|N/A|打上げ日時。現状このパラメータは使用していない。|
|Site|N/A|射場の選択。`1:大島_陸，1:大島_海，1:能代_陸，1:能代_海，5:任意の射場`。 `0`を選んだ場合，射点の絶対座標（緯度，経度，高度）を指定する。各射場の番号の対応は[こちらのファイル](input/key_launch_site.json)に準ずる。|
|Launch lat|deg|緯度|
|Launch lon|deg|経度|
|Launch height|deg|高度|
|Launch Azimuth|deg|打上げ方位角。磁北から反時計回りを正。|
|Launch Elevation|deg|打上げ仰角。|
|Launcher Rail Length |m|ランチャ有効レール長。|
|Tip-Off Calculation Exist|N/A|チップオフを考慮するかどうか。ガントリー式の場合，`false`にする。|
|Safety Area Exist|N/A|落下分散の出力に保安円を表示させたい場合`true`，いらない場合`false`|
|Input Magnetic Azimuth|deg|磁気偏角。磁北と真北のずれ。|

### Structure
構造に関するパラメータ。<br>
|項目|単位|備考|
|---|---|---|
|Length|m|機体全長。ノーズコーン先端から機体後端まで。ボートテイルを有する場合はボートテイル後端まで。ノズルカバーは含めない。|
|Diameter|m|機体代表直径。|
|Dry Mass|kg|乾燥時（酸化剤を除いた）の機体重量。|
|Dry Length-C.G. from Nosetip|m|乾燥時の機体重心とノーズコーン先端の距離。|
|Dry Moment of Inertia Roll-Axis|kg*m^2|乾燥時のロール軸回りの慣性モーメント。|
|Dry Moment of Inertia  Pitch-Axis|kg*m^2|乾燥時のピッチ軸回りの慣性モーメント。|
|Upper Launch Lug|m|1本目のランチラグとノーズコーン先端の距離。ランチラグが3本以上の場合，機体後端から2本目のランチラグとの距離にすること。|
|Lower Launch Lug|m|機体後端から最も近いランチラグとノーズコーン先端との距離。|

### Engine
エンジンに関するパラメータ。<br>
|項目|単位|備考|
|---|---|---|
|Thrust Curve|-|推力履歴のcsvファイルのパス。|
|Nozzle Exit Ratio|mm|ノズル出口（出力）径 |
|Burn Time|sec|燃焼時間。作動時間とは異なる。|
|Tank Volume|cc|酸化剤タンクの容量。|
|Oxidizer Density|kg/m^3|酸化剤密度。
|Length Fuel-C.G. from End|m|機体後端から燃料（グレイン，固形燃料）重心までの距離。インジェクターベルは燃料重心に含めない。|
|Length Tank-End from End|m|機体後端から酸化剤タンク口金までの距離。|
|Fuel Mass Before|kg|燃焼前燃料重量。|
|Fuel Mass After|kg|燃焼後燃料重量|
|Fuel Outside Diameter|mm|燃料外径。|
|Fuel Inside Diamter|mm|燃焼前の燃料内径。|
|Tank Diameter|mm|タンク外径。|
|Fuel Length|m|燃料長さ。インジェクターベル含めず。|
|Tank Length|m|タンク長さ。|

### Aero
空力微係数関連のパラメータ。<br>
|項目|単位|備考|
|---|---|---|
|Cd File|N/A|抗力係数（軸力係数）カーブのファイルのパス。|
|Cd File Exist|N/A|抗力係数カーブを使用する（抗力係数のマッハ数依存性を考慮する。）かどうか|
|Constant Cd|-|一定抗力係数。|
|Length-C.P. File|N/A|圧力中心カーブのファイルのパス。|
|Length-C.P. File Exist|N/A|圧力中心カーブを使用するかどうか。|
|Constant Length-C.P. from Nosetip|m|一定圧力中心。ノーズコーン先端からの距離。|
|CNa File|N/A|法線力係数傾斜（法線力傾斜。OpenRocketでいうところの法線力係数。）カーブ。|
|CNa File Exist|N/A|法線力係数傾斜カーブを使用するかどうか。|
|Constant CNa|1/rad|一定法線力係数傾斜。|
|Roll Dumping Moment Coefficient Clp|1/rad|ロール減衰モーメント係数。|
|Pitch Dumping Moment Coefficient Cmq|1/rad|ピッチ・ヨー減衰モーメント係数。|

### Parachute
パラシュート，ドローグシュートについてのパラメータ。<br>
|項目|単位|備考|
|---|---|---|
|1st Parachute CdS|m^2|1段目のパラシュート（単段分離ならメインシュート，2段分離ならドローグシュート）の抗力係数とパラシュート面積の積。|
|Parachute Opening Lag|sec|1段目パラシュートの開傘ラグ．開傘ラグを考慮しない場合，0.0にする．|
|2nd Parachute Exist|N/A|2段分離を行うかどうか。|
|2nd Parachute CdS|m2|2段目パラシュートのCdS|
|2nd Parachute Opening Altitude|m|2段目開傘の高度。|
|2nd Parachute Timer Mode|N/A|2団目パラシュートの開傘条件を時間で指定する場合`true`|
|2nd Timer [s]|sec|2団目パラシュート開傘時間。|

### Wind
風についての設定。<br>
|項目|単位|備考|
|---|---|---|
|Wind File Exist|N/A|上空風データ（csvファイル）を用いてシミュレーションを行うかどうか。|
|Wind File|N/A|上空風データのパス。|
|Wind Model|N/A|風モデルの指定。`Wind File Exist`が`false`の場合，有効となる。`law`,`constant`の中から選択。モデルの指定に誤りがあるか，指定なしの場合`constant`になる。|
|Wind Power Law Coefficient|-|高度分布係数。陸打ちは4.5，海打ちは6.0とする。運営から指定がある場合，そちらに従う。|
|Wind Speed|m/s|基準高度（風速計設置高度）での基準風速。`Simulation Mode`が`single`の場合のみ有効。|
|Wind Azimuth|deg|風向。北を0 deg，時計回り正。|
|Wind Reference Altitude|m|基準高度。風向風速計の設置高度とする。|

`Wind Model`については以下の通り。
|項目|説明|
|---|---|
|`law`|べき法則で風速を計算。風向は高度分布をもたない。|
|`constant`|定常風。風向・風速が高度分布をもたない。|

### Payload
|項目|説明|
|--|--|
|`Payload Exist`|ペイロードの分離を行うかどうか|
|`Mass [kg]`|ペイロードの重量。分離前はロケット側の構造質量に含まれ，分離後は燃焼終了時の質量からこの値がまるまる差し引かれる。|
|`Parachute CdS [m2]`|ペイロードのパラシュートの抗力特性|

# Other Information
* 射場情報の更新，マップ更新，射場追加は[こちら](PlotLandingScatter/launch_site/README.md)
* シミュレータの仕様，支配方程式の解法などは[こちら](src/quabla/simulator/README.md)

# Future Works
<!-- * 審査書の諸元表から諸元ファイル作成 -->
<!-- * 入力諸元のプレチェック実施（パラメータの不整合） -->
<!-- * 入力された推力履歴から推力解析の実施 -->
* ForRocket用の諸元ファイル自動生成
* 変数が発散したときの例外処理
* README加筆(プログラム構造回り)
* 作動時間の最大推力95%判定
* 真空中推力での質量減少率算出
* コマンドライン引数の受付
   * gui, cuiの切り替え
   * ヘルプコマンド
* guiツール
   * single, multiモードのコマンドボタン
   * 主要ファイル，パラメータの途中変更
   * モードに合わせたフォルダ名変更
* 審査書用の計算一通り実行するモード