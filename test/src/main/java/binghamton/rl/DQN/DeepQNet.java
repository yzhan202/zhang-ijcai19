package binghamton.rl.DQN;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;


public class DeepQNet<NN extends DeepQNet> implements IDeepQNet<NN> {
	
	final protected MultiLayerNetwork mln;
	boolean isRecurrent;
	
	public DeepQNet(MultiLayerNetwork mln) {
		this.mln = mln;
		this.isRecurrent = true;
	}


	@Override
	public NeuralNetwork[] getNeuralNetworks() {
		// TODO Auto-generated method stub
        return new NeuralNetwork[] { mln };
	}


	@Override
	public void save(OutputStream os) throws IOException {
		// TODO Auto-generated method stub
		ModelSerializer.writeModel(mln, os, true);
	}


	@Override
	public void save(String filename) throws IOException {
		// TODO Auto-generated method stub
		ModelSerializer.writeModel(mln, filename, true);
	}

	public static DeepQNet load(String path) throws IOException {
		return new DeepQNet(ModelSerializer.restoreMultiLayerNetwork(path));
	}

	@Override
	public boolean isRecurrent() {
		// TODO Auto-generated method stub
		return false;
	}


	@Override
	public void reset() {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void fit(INDArray input, INDArray labels) {
		// TODO Auto-generated method stub
		mln.fit(input, labels);
	}


	@Override
	public void fit(INDArray input, INDArray[] labels) {
		// TODO Auto-generated method stub
		fit(input, labels[0]);
	}


	@Override
	public INDArray output(INDArray batch) {
		// TODO Auto-generated method stub
//		if (isRecurrent)
//			return mln.output(batch);
//		else
//			return mln.rnnTimeStep(batch);
		return mln.output(batch);
	}


	@Override
	public INDArray[] outputAll(INDArray batch) {
		// TODO Auto-generated method stub
//		if (isRecurrent) {
//			return new INDArray[] {mln.rnnTimeStep(batch)};
//		} else {
//			return new INDArray[] {mln.output(batch)};
//		}
		return null;
	}


	@Override
	public NN clone() {
		// TODO Auto-generated method stub
		NN nn = (NN)new DeepQNet(mln.clone());
		nn.mln.setListeners(mln.getListeners());
		return nn;
	}


	@Override
	public void copy(NN from) {
		// TODO Auto-generated method stub
		mln.setParams(from.mln.params());
	}


	@Override
	public Gradient[] gradient(INDArray input, INDArray label) {
		// TODO Auto-generated method stub
		mln.setInput(input);
		mln.setLabels(label);
		mln.computeGradientAndScore();
		Collection<TrainingListener> iterationListeners = mln.getListeners();
		if (iterationListeners != null && iterationListeners.size() >0) {
			for (TrainingListener l : iterationListeners) {
				l.onGradientCalculation(mln);
			}
		}
		System.out.println("SCORE: "+ mln.score());
		return new Gradient[] {mln.gradient()};
	}


	@Override
	public Gradient[] gradient(INDArray input, INDArray[] label) {
		// TODO Auto-generated method stub
		return gradient(input, label[0]);
	}


	@Override
	public void applyGradient(Gradient[] gradient, int batchSize) {
		// TODO Auto-generated method stub
		MultiLayerConfiguration mlnConf = mln.getLayerWiseConfigurations();
		int iterationCount = mlnConf.getIterationCount();
		int epochCount = mlnConf.getEpochCount();
		mln.getUpdater().update(mln, gradient[0], iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
		mln.params().subi(gradient[0].gradient());
		Collection<TrainingListener> iterationListeners = mln.getListeners();
		if (iterationListeners != null && iterationListeners.size() >0) {
			for (TrainingListener listener : iterationListeners) {
				listener.iterationDone(mln, iterationCount, epochCount);
			}
		}
		mlnConf.setIterationCount(iterationCount+ 1);
	}


	@Override
	public double getLatestScore() {
		// TODO Auto-generated method stub
		return mln.score();
	}


	@Override
	public void save(String pathValue, String pathPolicy) throws IOException {
		// TODO Auto-generated method stub
		
	}
	

}












