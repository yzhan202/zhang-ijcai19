package binghamton.rl.A3C;

import org.nd4j.linalg.api.ndarray.INDArray;


public class MiniTrans {
	public INDArray obs;
	public Integer action;
	public INDArray valueOutput;
	public double reward;
	
	public MiniTrans(INDArray obs, Integer action, INDArray valueOutput, double reward) {
		this.obs = obs;
		this.action = action;
		this.valueOutput = valueOutput;
		this.reward = reward;
	}
}





