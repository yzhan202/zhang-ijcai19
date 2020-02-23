package binghamton.rl.DQN;

import org.nd4j.linalg.api.ndarray.INDArray;


public class StateActTrans {
	
	public INDArray observation;
	public int action; // <row, col>
	public double reward;
	public boolean isTerminal;
	public INDArray nextObservation;
	
	public StateActTrans(INDArray observation, int action, double reward, boolean isTerminal, INDArray nextObservation) {
		this.observation = observation;
		this.action = action;
		this.reward = reward;
		this.isTerminal = isTerminal;
		this.nextObservation = nextObservation;
	}
	
	public StateActTrans dup() {
		INDArray dupObservation = observation.dup();
		INDArray nextObs = nextObservation.dup();
		return new StateActTrans(dupObservation, action, reward, isTerminal, nextObs);
	}
	
	public double getReward() {
		return reward;
	}
}



