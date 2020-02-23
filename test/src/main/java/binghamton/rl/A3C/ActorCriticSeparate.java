package binghamton.rl.A3C;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import binghamton.rl.DQN.DeepQNet;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;


public class ActorCriticSeparate<NN extends ActorCriticSeparate> implements IActorCritic<NN> {

	final protected MultiLayerNetwork valueNet;
	final protected MultiLayerNetwork policyNet;
	final protected boolean recurrent;
	
	public ActorCriticSeparate(MultiLayerNetwork valueNet, MultiLayerNetwork policyNet) {
		this.valueNet = valueNet;
		this.policyNet = policyNet;
		this.recurrent = valueNet.getOutputLayer() instanceof RnnOutputLayer;
	}
	
	@Override
	public NeuralNetwork[] getNeuralNetworks() {
		// TODO Auto-generated method stub
		return new NeuralNetwork[] {valueNet, policyNet};
	}

	@Override
	public void save(OutputStream os) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void save(String filename) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean isRecurrent() {
		// TODO Auto-generated method stub
		return recurrent;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		if (recurrent) {
			valueNet.rnnClearPreviousState();
			policyNet.rnnClearPreviousState();
		}
	}

	@Override
	public void fit(INDArray input, INDArray[] labels) {
		// TODO Auto-generated method stub
		valueNet.fit(input, labels[0]);
		policyNet.fit(input, labels[1]);
	}

	@Override
	public INDArray[] outputAll(INDArray batch) {
		// TODO Auto-generated method stub
		if (recurrent) {
			return new INDArray[] {valueNet.rnnTimeStep(batch), policyNet.rnnTimeStep(batch)};
		} else {
			return new INDArray[] {valueNet.output(batch), policyNet.output(batch)};
		}
	}

	@Override
	public NN clone() {
		// TODO Auto-generated method stub
		NN nn = (NN)new ActorCriticSeparate(valueNet.clone(), policyNet.clone());
		nn.valueNet.setListeners(valueNet.getListeners());
		nn.policyNet.setListeners(policyNet.getListeners());
		return nn;
	}

	@Override
	public void copy(NN from) {
		// TODO Auto-generated method stub
		valueNet.setParams(from.valueNet.params());
		policyNet.setParams(from.policyNet.params());
	}

	@Override
	public Gradient[] gradient(INDArray input, INDArray[] labels) {
		// TODO Auto-generated method stub
		valueNet.setInput(input);
		valueNet.setLabels(labels[0]);
		valueNet.computeGradientAndScore();
		Collection<TrainingListener> iterationListeners = valueNet.getListeners();
		if (iterationListeners != null && iterationListeners.size() >0) {
			for (TrainingListener l : iterationListeners) {
				l.onGradientCalculation(valueNet);
			}
		}

        policyNet.setInput(input);
        policyNet.setLabels(labels[1]);
        policyNet.computeGradientAndScore();
        Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (TrainingListener l : policyIterationListeners) {
                l.onGradientCalculation(policyNet);
            }
        }   
        MultiLayerConfiguration valueConf = valueNet.getLayerWiseConfigurations();
        int valueIterationCount = valueConf.getIterationCount();
        valueConf.setIterationCount(valueIterationCount + 1);
        
        return new Gradient[] {valueNet.gradient(), policyNet.gradient()};
	}

	@Override
	public void applyGradient(Gradient[] gradient, int batchSize) {
		// TODO Auto-generated method stub
		MultiLayerConfiguration valueConf = valueNet.getLayerWiseConfigurations();
		int valueIterationCount = valueConf.getIterationCount();
		int valueEpochCount = valueConf.getEpochCount();
		valueNet.getUpdater().update(valueNet, gradient[0], valueIterationCount, valueEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
		valueNet.params().subi(gradient[0].gradient());
		Collection<TrainingListener> valueIterationListeners = valueNet.getListeners();
		if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
		    for (TrainingListener listener : valueIterationListeners) {
		        listener.iterationDone(valueNet, valueIterationCount, valueEpochCount);
		    }
		}
		valueConf.setIterationCount(valueIterationCount + 1);
		
		MultiLayerConfiguration policyConf = policyNet.getLayerWiseConfigurations();
		int policyIterationCount = policyConf.getIterationCount();
		int policyEpochCount = policyConf.getEpochCount();
		policyNet.getUpdater().update(policyNet, gradient[1], policyIterationCount, policyEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
		policyNet.params().subi(gradient[1].gradient());
		Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
		if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
		    for (TrainingListener listener : policyIterationListeners) {
		        listener.iterationDone(policyNet, policyIterationCount, policyEpochCount);
		    }
		}
		policyConf.setIterationCount(policyIterationCount + 1);
		
//		System.out.println("Policy Net Gradient: "+ gradient[1].gradient());
	}

	@Override
	public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void save(String pathValue, String pathPolicy) throws IOException {
		// TODO Auto-generated method stub
		ModelSerializer.writeModel(valueNet, pathValue, true);
        ModelSerializer.writeModel(policyNet, pathPolicy, true);
	}

	@Override
	public double getLatestScore() {
		// TODO Auto-generated method stub
		return valueNet.score();
	}
	
	public static ActorCriticSeparate load(String valePath, String policyPath) throws IOException {
		return new ActorCriticSeparate(ModelSerializer.restoreMultiLayerNetwork(valePath), ModelSerializer.restoreMultiLayerNetwork(policyPath));
	}
	
}




