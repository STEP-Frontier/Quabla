package quabla;

import java.io.File;
import java.io.IOException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import quabla.simulator.MultiSolver;
import quabla.simulator.Solver;
import quabla.simulator.rocket.Rocket;

public class QUABLA {

	public static String simulationModeCheck;

	public static void main(String[] args) throws IOException {

		String line = args[0];

		String simulationMode;
		String filepathResult;

		while(true) {
			simulationModeCheck = args[1];
			if(simulationModeCheck.equals("single")) {
				simulationMode = "single";
				break;
			}
			else if(simulationModeCheck.equals("multi")) {
				simulationMode = "multi";
				break;
			}
			else {
				System.out.println("Please input single or multi.");
				System.exit(0);
			}
		}

		System.out.println("[Solver] Running...");

		ObjectMapper mapper = new ObjectMapper();
		JsonNode node = null;
		try {
			node = mapper.readTree(new File(line)); //諸元のjsonファイル名を入れる
		}catch(IOException e) {
			e.printStackTrace();
		}

		long startTime = System.currentTimeMillis();

		switch(simulationMode) {
		case "single":
			//single condition

			filepathResult = args[2];
			String filepathResultFinal = filepathResult + File.separator;

			Rocket rocket = new Rocket(node);
			rocket.outputSpec(filepathResultFinal, simulationMode);
			Solver solver = new Solver(filepathResultFinal);
			solver.solveDynamics(rocket);		// 飛翔軌道メイン計算
			solver.makeResult();				// 計算結果出力
			solver.outputResultTxt();			// 計算結果サマリー出力
			solver.outputResultSummary();		// Pythonに渡す用のイベント値出力
			solver.outputLandPoint();			// 落下地点出力
			break;

		case "multi":
			//multiple condition

			String filepathResultMulti = args[2];
			String filepathResultMultiFinal = filepathResultMulti + File.separator;
			int procCpuMax = Integer.parseInt(args[3]);	// 並列計算のプロセス数取得

			MultiSolver multiSolver = new MultiSolver(filepathResultMultiFinal, node.get("Multi Solver"), node.get("Payload").get("Payload Exist").asBoolean(), procCpuMax);
			multiSolver.solveMulti(node);				// メイン計算
			break;
		}

		long endTime = System.currentTimeMillis();
		System.out.println(String.format("[Solver] Calculate Time : %.3f sec", (endTime - startTime) * 1e-03));
		System.out.println("[Solver] Done!");
	}
}