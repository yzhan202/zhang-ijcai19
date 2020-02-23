package binghamton.rl_latent.A3C.generator_latent;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jcodec.common.ArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.omg.CORBA.Current;

import binghamton.rl.A3C.ActorCriticLoss;
import binghamton.rl.A3C.ActorCriticSeparate;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;

import binghamton.rl_latent.A3C.latentPSLModelCreation;


public class a3cMDP_latent {
	final int maxEpochStep;
	final int numThread;
	final int nstep; //t_max
	final double gamma;
	
	final int maxRuleLen;
	final int maxRuleNum;
		
//	ActorCriticSeparate actorCriticNN;
	AsyncGlobal_latent asyncGlobal;
	
//	StandardPredicate UserDummyPred; // attendsAA(U)
//	StandardPredicate FriendDummyPred; // friend(U,U1)
	
	public a3cMDP_latent() {
		maxRuleLen = 3; //6; 5;  
		maxRuleNum = 12; // 8; 12; 15; 18
		
		maxEpochStep = (int)3.0e+5; // 5e+5
		numThread = 2; //2; 3; 
		nstep = 6; // 6, 
		gamma = 1.0; //0.99;
		
		latentPSLModelCreation pslGenerator = new latentPSLModelCreation(123456);
		StandardPredicate[] X = pslGenerator.getX();
		StandardPredicate[] Y = pslGenerator.getY();
		StandardPredicate[] Z = pslGenerator.getZ();
		
		// Dummy Predicates
		StandardPredicate UserDummyPred = (StandardPredicate)PredicateFactory.getFactory().getPredicate("questionAnswer");
		X = ArrayUtils.removeElement(X, UserDummyPred);
		
		StandardPredicate[] allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(X, Y), Z);
		StandardPredicate[] negFeatPreds = pslGenerator.getNegFeatPreds();
		StandardPredicate[] allNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, Y), Z);
//		if (PredNum < maxRuleBodyLen)
//			throw new IllegalArgumentException("Max rule body length cannot bigger than Predicates Number");
		int posPredNum = allPosPreds.length;
		int negPredNum = allNegPreds.length;
		
		final int outputSize = posPredNum+ negPredNum+1; // Positive and Negative Predicates and "Return"
		final int inputRow = maxRuleNum;
		final int inputCol = outputSize; // Positive, Negative Predicates and "Return"
				
		ActorCriticSeparate actorCriticNN = buildActorCritic(new int[] {inputRow, inputCol}, outputSize);
		asyncGlobal = new AsyncGlobal_latent(actorCriticNN, maxEpochStep, maxRuleLen, maxRuleNum, gamma);
	}
	
	public ActorCriticSeparate buildActorCritic(int[] numInputs, int numOutputs) {				
		final int NEURAL_NET_SEED = 12345;
		final double l2 = 0; //1e-5; //0.0001;
		final int numHiddenNodes = 16; // 18; 16
		final int numLayers = 3;
		final boolean isUseLSTM = false; //true;
		final int NEURAL_NET_ITERATION_LISTENER = 5000;
		final double adam = 1e-3; // 0.001
		
		int nIn = 1;
		for (int i: numInputs) {
			nIn *= i;
		}
		System.out.println("nIn: "+ nIn);
		
		/*
		 * Value Net
		 */
		NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(NEURAL_NET_SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(adam))
				.weightInit(WeightInit.XAVIER)
				.l2(l2)
				.list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
					.activation(Activation.TANH).build()); //RELU
		for (int i=1; i<numLayers; i++) {
			confB.layer(i, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .activation(Activation.TANH).build());
		}	
		confB.layer(numLayers, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                .nIn(numHiddenNodes).nOut(1).build());
		
		confB.setInputType(InputType.feedForward(nIn));
		MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork valueNet_model = new MultiLayerNetwork(mlnconf);
        valueNet_model.init();
        valueNet_model.setListeners(new ScoreIterationListener(NEURAL_NET_ITERATION_LISTENER));
		
        /*
         * Policy Net
         */
        NeuralNetConfiguration.ListBuilder confB2 = new NeuralNetConfiguration.Builder().seed(NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(adam))
                .weightInit(WeightInit.XAVIER)
//                .regularization(true)
                .l2(l2)
                .list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
                                .activation(Activation.TANH).build()); //RELU
        for (int i = 1; i < numLayers; i++) {
            confB2.layer(i, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                            .activation(Activation.TANH).build());
        }
        confB2.layer(numLayers, new OutputLayer.Builder(new ActorCriticLoss())
                        .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build());
        
        confB2.setInputType(InputType.feedForward(nIn));
        MultiLayerConfiguration mlnconf2 = confB2.pretrain(false).backprop(true).build();
        MultiLayerNetwork policyNet_model = new MultiLayerNetwork(mlnconf2);
        policyNet_model.init();
        policyNet_model.setListeners(new ScoreIterationListener(NEURAL_NET_ITERATION_LISTENER));
        
		return new ActorCriticSeparate(valueNet_model, policyNet_model);
	}
	
	public AsyncThread_latent newThread(int i) {
		return new AsyncThread_latent(asyncGlobal, i, maxRuleLen, maxRuleNum, nstep, gamma);
	}
	
	protected boolean isTrainingComplete() {
		return asyncGlobal.isTrainingComplete();
	}
	
	public int getStepCounter() {
		return asyncGlobal.getT().get();
	}
	
	public void train() {
		try {
			System.out.println("AsyncLearning training Starting.");
			asyncGlobal.start();
			
			for (int i=0; i<numThread; i++) {
				Thread t = newThread(i);
				Nd4j.getAffinityManager().attachThreadToDevice(t, i % Nd4j.getAffinityManager().getNumberOfDevices());
				t.start();
				
//				AsyncThread t = newThread(i);
//				t.run();
			}
			
			asyncGlobal.join();
		} catch (Exception e) {
			System.out.println("Training Failed. "+ e);
			e.printStackTrace();
		}
	}
	
	public void saveA3C(String path1, String path2) throws IOException {
		asyncGlobal.current.save(path1, path2);
	}
	
	public void loadA3C(String path0, String path1) throws IOException {
		asyncGlobal.current = ActorCriticSeparate.load(path0, path1);
	}
}



















