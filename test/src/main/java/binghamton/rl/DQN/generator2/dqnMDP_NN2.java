package binghamton.rl.DQN.generator2;

import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Term;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.atom.Atom;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.formula.Formula;
import edu.umd.cs.psl.model.formula.Rule;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.util.database.Queries;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java. util. Iterator;
import java.util.Random;
import java.util.Scanner;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.util.Log;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import com.google.common.collect.Iterables;

import binghamton.rl.DQN.DeepQNet;
import binghamton.rl.DQN.ExperienceReplay;
import binghamton.rl.DQN.StateActTrans;

import org.apache.commons.lang3.ArrayUtils;

import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication;
import edu.umd.cs.psl.application.learning.weight.em.HardEM;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;


public class dqnMDP_NN2 {
	
	final int offset; //num step for eps greedy anneal
	final double maxAlpha; 
	final double minAlpha; //min epsilon
	
	final StandardPredicate[] X;
	final StandardPredicate[] Y;
	final StandardPredicate[] Z;
	final StandardPredicate[] allPosPreds;
	final StandardPredicate[] allNegPreds;
	
	final StandardPredicate[] friendPreds;
	final StandardPredicate[] networkPreds;

	final Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	final Map<StandardPredicate, List<List<Object>>> specificPredArgsMap;
	final int maxRuleLen;
	final int maxRuleNum;
	final int posPredNum;
	final int negPredNum;
	
	PSLModel model;
	DataStore data;
	Database wlTruthDB;
	Partition trainPart;
	
	int[][] ruleListEmbedding;
	
	final int maxEpisode;
	final int batchSize;
//	int maxStep;
	private DeepQNet currentDQN;
	private DeepQNet targetDQN;
	
	final double GAMMA; //Discount Factor
	final int targetDqnUpdateFreq; //target update (hard)
	final int expRepMaxSize; //Max size of experience replay
	final int NEURAL_NET_SEED = 12345;
	Random rdm = new Random(NEURAL_NET_SEED);
    final int NEURAL_NET_ITERATION_LISTENER = 10000;
    final int updateStart; // 10
    
	ExperienceReplay expReplay; 
	
	final int inputRow;
	final int inputCol;
	final int outputSize;
	
	final int ReturnAction;
	
	final double LAMBDA_ruleLen = 0.0;//0.5;
	final double LAMBDA_ruleNum = 0.0;//0.5;
	final double LAMBDA_coverage = 0.0;//0.2;
	final double MIN_VAL = 1e-6;
	
	private ConfigBundle config;
	final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
	final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};
	
	public dqnMDP_NN2(StandardPredicate[] X, StandardPredicate[] Y, StandardPredicate[] Z, StandardPredicate[] negFeatPreds,
			PSLModel model, DataStore data, Database wlTruthDB, Partition trainPart, ConfigBundle config,
			Map<StandardPredicate,List<GenericVariable>> generalPredArgsMap, Map<StandardPredicate,List<List<Object>>> specificPredArgsMap,
			StandardPredicate[] friendPreds, StandardPredicate[] networkPreds) {
		this.maxRuleLen = 5; // 6, 4
		this.maxRuleNum = 8;
		
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		this.allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(this.X, this.Y), this.Z);
		this.allNegPreds = allPosPreds.clone(); //ArrayUtils.addAll(negFeatPreds, this.Y);
		this.posPredNum = allPosPreds.length; //X.length+ Y.length+ Z.length;
		this.negPredNum = allNegPreds.length; //posPredNum; // negFeatPreds.length+ 1;
		this.generalPredArgsMap = generalPredArgsMap;
		this.specificPredArgsMap = specificPredArgsMap;
		this.friendPreds = friendPreds;
		this.networkPreds = networkPreds;
		
		this.model = model;
		this.data = data;
		this.wlTruthDB = wlTruthDB;
		this.trainPart = trainPart;
				
		this.outputSize = posPredNum+ negPredNum+ 1; // Positive and Negative Predicates and "Return"
		this.inputRow = maxRuleNum;
		this.inputCol = outputSize; // Positive, Negative Predicate and "Return"
 		
		this.ruleListEmbedding = new int[inputRow][inputCol];
		
		this.maxEpisode = (int)1.8e+4;
		this.batchSize = 32; // 32;
		this.updateStart = 16; 
		
		this.offset = 2000; // 2000; 2500; 3000
		this.maxAlpha = 1.0;
		this.minAlpha = 0.1;
		
		currentDQN = buildDenseDQN(new int[] {inputRow, inputCol}, outputSize);
		targetDQN = currentDQN.clone();
		
		GAMMA = 1.0; //0.99;
		targetDqnUpdateFreq = 500; // 500; 200
		expRepMaxSize = 2000; // 500; 2000; 1000
		expReplay = new ExperienceReplay(expRepMaxSize, batchSize, rdm);
				
		ReturnAction = outputSize-1;
		
//		REWARD_THRESHOLD = 0; //calculateThreshold();
		this.config = config;
	}
	
	void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
		Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
		DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);
		
		for (int i=0; i<Y.length; i++)
			dbPop.populateFromDB(populationDatabase, Y[i]);

		populationDatabase.close();
	}
	
	public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        for (int i = 0; i < shape.length; i++) {
            nshape[i + 1] = shape[i];
        }
        return nshape;
    }

    public static int[] makeShape(int batch, int[] shape, int length) {
        int[] nshape = new int[3];
        nshape[0] = batch;
        nshape[1] = 1;
        for (int i = 0; i < shape.length; i++) {
            nshape[1] *= shape[i];
        }
        nshape[2] = length;
        return nshape;
    }
	
	public void saveDQN(String savePath) throws IOException {
//		String savePath = "/home/yue/Public/Java/structureLearning4PSL/test/result/dqn.ser";
		currentDQN.save(savePath);
	}
	
	
	public void loadDQN(String savePath) throws IOException {
		currentDQN = currentDQN.load(savePath);
		targetDQN = currentDQN.clone();
	}
    
	public double epsilonSchedule(int numVisit) {
		double epsilon = offset*1.0 / (offset+ numVisit);
		epsilon = Math.min(epsilon, maxAlpha);
		epsilon = Math.max(epsilon, minAlpha);
		return epsilon;
	}
	
	public void resetRuleListEmbedding() {
		IntStream.range(0,inputRow).forEach(i -> 
			IntStream.range(0, inputCol).forEach(j-> ruleListEmbedding[i][j]=0));
	}
	
	public DeepQNet buildDenseDQN(int numInputs[], int numOutputs) {
		int nIn = 1;
        for (int i : numInputs) {
            nIn *= i;
        }
        final int layerNum = 3;
        final int numHiddenNodes = 8; //8; 10;;
        final double l2 = 1e-5;
        final double adam = 1e-3; //1e-3; 2e-4;
       
        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(NEURAL_NET_SEED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Adam(adam))
                        .weightInit(WeightInit.XAVIER)
                        .l2(l2)
                        .list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
                        		.activation(Activation.TANH).build()); //RELU; TANH
        
        for (int i=1; i<layerNum; i++) {
        	confB.layer(i, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        					.activation(Activation.TANH).build()); //RELU
        }
        
        confB.layer(layerNum, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY) // IDENTITY
        		.nIn(numHiddenNodes).nOut(numOutputs).build());

        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork nn_model = new MultiLayerNetwork(mlnconf);
        nn_model.init();
        nn_model.setListeners(new ScoreIterationListener(NEURAL_NET_ITERATION_LISTENER));
        
        return new DeepQNet(nn_model);
	}
	
	public void updateTargetNetwork() {
		targetDQN = currentDQN.clone();
	}
	
	
	public double optimalSolution() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		int nextAccessIdx;
		int[] lastRulePreds = new int[posPredNum];
		int currentAct;
		double reward;
		double accumulatedReward=0;
		
		boolean END_SIGNAL;
		boolean RULE_END_SIGNAL;
			
		boolean TARGET_PENALTY_SIGNAL;
		boolean GRAMMAR_PENALTY_SIGNAL;
		boolean NETWORK_PENALTY_SIGNAL;
		
		/*
		 *  Initialization
		 */
		cleanPSLrule();
		resetRuleListEmbedding();
		nextAccessIdx = 0;
		IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);
		END_SIGNAL = false;
		RULE_END_SIGNAL = false;
		accumulatedReward = 0;
		
		TARGET_PENALTY_SIGNAL = false; // Check on each Rule
		GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
		NETWORK_PENALTY_SIGNAL = false; // Check on each Rule
		
		int countStep = 0;
		while(!END_SIGNAL) {
			
			/*
			 * Greedy Action
			 */			
			currentAct = nextAction();
			
			INDArray observation = processHistory();
			INDArray dqnOutput = currentDQN.output(observation).reshape(outputSize);
//			if (currentAct<posPredNum)
//				System.out.println(""+ countStep+ ": ["+ currentAct+ ","+ allPosPreds[currentAct]+ "], "+ dqnOutput);
//			else if (currentAct<ReturnAction)
//				System.out.println(""+ countStep+ ": ["+ currentAct+ ",~"+ allNegPreds[currentAct-posPredNum]+ "], "+ dqnOutput);
//			else
//				System.out.println(""+ countStep+ ": ["+ currentAct+ ",Return], "+ dqnOutput);
			countStep++;
			
			/*
			 * State-Action Transition
			 */
			if (!RULE_END_SIGNAL) {
				if (currentAct == ReturnAction) { // Rule Return
					ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
					RULE_END_SIGNAL = true;
				} else {
					int posPredIdx = getPosPredIndex(currentAct);
					lastRulePreds[posPredIdx] += 1;
					if (lastRulePreds[posPredIdx] == 1) 
						ruleListEmbedding[nextAccessIdx][currentAct] = 1;
					
					if (IntStream.of(lastRulePreds).sum() >= maxRuleLen) {
						ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
						RULE_END_SIGNAL = true;
					}
				}
			}
			if (RULE_END_SIGNAL && IntStream.of(lastRulePreds).sum()>0) {
				// Check Target
				if (lastRulePreds[posPredNum-1]==0)
					TARGET_PENALTY_SIGNAL = true;
				// Check Network Rule
				if (checkIsNetworkRule(lastRulePreds)) {
					NETWORK_PENALTY_SIGNAL = !checkValideNetworkRule(ruleListEmbedding[nextAccessIdx]);
				}
				// Check Grammar
				if (!TARGET_PENALTY_SIGNAL && !NETWORK_PENALTY_SIGNAL) {
					GRAMMAR_PENALTY_SIGNAL = !buildNewRule(ruleListEmbedding[nextAccessIdx]);
				}
			}
			
			/*
			 * Assign Reward
			 */
			reward = 0;
			if (TARGET_PENALTY_SIGNAL || GRAMMAR_PENALTY_SIGNAL || NETWORK_PENALTY_SIGNAL) {
				END_SIGNAL = true;
			}
			if (RULE_END_SIGNAL) {
				nextAccessIdx++;
				// Check Finishing Building Rule List
				if (!END_SIGNAL) {
					if ((nextAccessIdx == maxRuleNum) || (IntStream.of(lastRulePreds).sum()==0)) {
						END_SIGNAL = true;
					}
				}
			}
			if (END_SIGNAL) {
				int kernelSize = 0;
				for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
					kernelSize++;
				}
				if (kernelSize!=0) { // && checkContainsNegTargetInSeq()
//					// Clean Inference Database
//					cleanInferenceDatabase();
					double lossFunction_reward = 0; //REWARD_THRESHOLD;					
//					// Reward from Binary Classification Entropy
//					double tmp = BCEcost();
//					lossFunction_reward += tmp;
					double tmp1 = AUCcost();
					lossFunction_reward += tmp1;
					//Reward from Interpretability Constaints
//					double tmp2 = interpretabilityCost();
//					lossFunction_reward += tmp2;
					
					reward = lossFunction_reward; //Scaling* lossFunction_reward/REWARD_THRESHOLD;
				} else {
					reward = 0;
				}
			}
			accumulatedReward += reward;
			
			/*
			 * Reset Signals
			 */
			if (RULE_END_SIGNAL) {
				RULE_END_SIGNAL = false;
				IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);

				TARGET_PENALTY_SIGNAL = false;
				GRAMMAR_PENALTY_SIGNAL = false;
				NETWORK_PENALTY_SIGNAL = false;
			}
		}
//		System.out.println(model.toString());
//		System.out.println("Area under ROC curve: "+ accumulatedReward);
		
		return accumulatedReward;
	}

	
	public double[] training() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		double bestScore = 0;
		String optimalRuleList = "";
		int bestEpochIdx = 0;
		
		int nextAccessIdx;
		int[] lastRulePreds = new int[posPredNum];
		int currentAct;
		int stepCounter = 0;
		double alpha=1.0;
		double reward;
		int kernelSize = 0;
		
		double accumulatedReward;
		
		boolean END_SIGNAL;
		boolean RULE_END_SIGNAL;
		
		boolean REDUNDANT_PENALTY_SIGNAL;
		boolean TARGET_PENALTY_SIGNAL;
		boolean GRAMMAR_PENALTY_SIGNAL;
		boolean NETWORK_PENALTY_SIGNAL;
		
		double[] trend = new double[maxEpisode];
		for (int epoch=0; epoch<maxEpisode; epoch++) {
			/*
			 *  Initialization
			 */
			cleanPSLrule();
			resetRuleListEmbedding();
			nextAccessIdx = 0;
			IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);
			END_SIGNAL = false;
			RULE_END_SIGNAL = false;
			accumulatedReward = 0;
			
			REDUNDANT_PENALTY_SIGNAL = false; // check on each Predicate
			TARGET_PENALTY_SIGNAL = false; // Check on each Rule
			GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
			NETWORK_PENALTY_SIGNAL = false; // Check on each RUle
			
			alpha = epsilonSchedule(epoch);
			while(!END_SIGNAL) {
//				REDUNDANT_PENALTY_SIGNAL = false;
//				alpha = epsilonSchedule(stepCounter);
				
				INDArray observation = processHistory();
				if (rdm.nextDouble() < alpha) {
					/*
					 * Exploration
					 * Random Action
					 */
					currentAct = rdm.nextInt(outputSize);
				} else {
					/*
					 *  Exploitation
					 *  Greedy Action
					 */
					currentAct = nextAction();
				}
				
				/*
				 * State-Action Transition
				 */
				if (!RULE_END_SIGNAL) {
					if (currentAct == ReturnAction) { // Rule Return
						ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
						RULE_END_SIGNAL = true;
					} else {
						int posPredIdx = getPosPredIndex(currentAct);
						lastRulePreds[posPredIdx] += 1;
						if (lastRulePreds[posPredIdx] == 1) 
							ruleListEmbedding[nextAccessIdx][currentAct] = 1;
						
						if (IntStream.of(lastRulePreds).sum() >= maxRuleLen) {
							ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
							RULE_END_SIGNAL = true;
						}
					}
				}
				if (RULE_END_SIGNAL && IntStream.of(lastRulePreds).sum()>0) { // && IntStream.of(lastRulePreds).sum()>0
					// Check Target
					if (lastRulePreds[posPredNum-1]==0)
						TARGET_PENALTY_SIGNAL = true;
					// Check Network Rule
					if (checkIsNetworkRule(lastRulePreds)) {
						NETWORK_PENALTY_SIGNAL = !checkValideNetworkRule(ruleListEmbedding[nextAccessIdx]);
					}
					// Check Grammar
					if (!TARGET_PENALTY_SIGNAL && !NETWORK_PENALTY_SIGNAL) {
						GRAMMAR_PENALTY_SIGNAL = !buildNewRule(ruleListEmbedding[nextAccessIdx]);
					}
				}
				
				/*
				 * Assign Reward
				 */
				reward = 0;
				if (TARGET_PENALTY_SIGNAL || GRAMMAR_PENALTY_SIGNAL || NETWORK_PENALTY_SIGNAL) {
					END_SIGNAL = true;
				}
				if (RULE_END_SIGNAL) {
					nextAccessIdx++;
					// Check Finishing Building Rule List
					if (!END_SIGNAL) {
						if ((nextAccessIdx == maxRuleNum) || (IntStream.of(lastRulePreds).sum()==0)) {
							END_SIGNAL = true;
						}
					}
				}
				if (END_SIGNAL) {
					kernelSize = 0;
					for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
						kernelSize++;
					}
					if (kernelSize!=0 && checkContainsNegTargetInSeq()) { // && checkContainsNegTargetInSeq()
//						// Clean Inference Database
//						cleanInferenceDatabase();
						double lossFunction_reward = 0; //REWARD_THRESHOLD;					
//						// Reward from Binary Classification Entropy
//						double tmp = BCEcost();
//						lossFunction_reward += tmp;
						double tmp1 = AUCcost();
						lossFunction_reward += tmp1;
						//Reward from Interpretability Constaints
//						double tmp2 = interpretabilityCost();
//						lossFunction_reward += tmp2;
						
						reward = lossFunction_reward; //Scaling* lossFunction_reward/REWARD_THRESHOLD;
					} else {
						reward = 0;
					}
				}
				accumulatedReward += reward;
				
				/*
				 * Store to Experience Replay
				 */
				INDArray nextObservation = processHistory();
				
				StateActTrans transition = new StateActTrans(observation, currentAct, reward, END_SIGNAL, nextObservation); 
				expReplay.store(transition);
				
				/*
				 * Sample Random minibatch
				 */
				if (stepCounter > updateStart) {
					ArrayList<StateActTrans> transBatch = expReplay.getBatch();
					Pair<INDArray, INDArray> target_data = setTarget(transBatch);	
					
//					INDArray output = currentDQN.output(target_data.getFirst());
					
					currentDQN.fit(target_data.getFirst(), target_data.getSecond());
					
//					INDArray afterOutput = currentDQN.output(target_data.getFirst());
//					for (int t=0; t<transBatch.size(); t++) {
//						StateActTrans trans = transBatch.get(t);
//						int p = trans.action;
//						double value = target_data.getSecond().getDouble(t,p);
//						double valueBefore = output.getDouble(t,p);
//						double valueHat = afterOutput.getDouble(t,p);
////						if (Math.abs(value - 0.55) < 0.1) {
////							System.out.println(""+ t+ " ["+ value + ": ["+ valueBefore+ ", "+ valueHat+"]");
////						}
//						if (value < valueBefore) {
//							System.out.println(""+ t+ " ["+ value + ": ["+ valueBefore+ ", "+ valueHat+"]");
//						}
//					}
				}
				
				stepCounter++;
				if (stepCounter % targetDqnUpdateFreq == 0) {
					updateTargetNetwork();
				}
				
				/*
				 * Reset Signals
				 */
				if (RULE_END_SIGNAL) {
					RULE_END_SIGNAL = false;
					IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);

					TARGET_PENALTY_SIGNAL = false;
					GRAMMAR_PENALTY_SIGNAL = false;
					NETWORK_PENALTY_SIGNAL = false;
				}
				if (END_SIGNAL) {
					if (accumulatedReward > bestScore) {
						bestScore = accumulatedReward;
						optimalRuleList = model.toString();
						bestEpochIdx = epoch;
						System.out.println("\nUpdate Optimal Solution: "+ bestScore+ ", "+ optimalRuleList);
					}
				}
			}
			
			trend[epoch] = accumulatedReward;
			if (epoch % 50 ==0) {
				System.out.println("Epoch: "+ epoch+ ", Step Counter: "+ stepCounter+ 
						", Alpha: "+ alpha+ ", Cummulated Reward: "+ accumulatedReward+ ", Kernel Size: "+ kernelSize);
				if (kernelSize > 0) {
					System.out.println(model.toString());
//					System.out.println("Best Score: "+ bestScore+ " at Epoch "+bestEpochIdx);
					System.out.println("Optimal Solution: "+ optimalSolution());
//					for (int r=0; r<inputRow; r++) {
//						for (int c=0; c<inputCol; c++) {
//							System.out.print(""+ ruleListEmbedding[r][c]+ " ");
//						}
//						System.out.println("");
//					}
				}
			}
		}
		System.out.println("Best Score: "+ bestScore+ " at Epoch "+bestEpochIdx);
		System.out.println(optimalRuleList);
		return trend;
	}
	
	public void cleanPSLrule() {
		Iterator<Kernel> kernels = model.getKernels().iterator();
		List<Kernel> kernelList = new ArrayList<>();
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
			kernelList.add(k);
		}
		for (int i=0; i<kernelList.size(); i++) {
			model.removeKernel(kernelList.get(i));
		}
	}
	
	
	public int getPosPredIndex(int currentAct) {
		if (currentAct==ReturnAction)
			return -1;
		
		int predIdx = -1;
		if (currentAct < posPredNum) { // Positive Predicate
			return currentAct;
		} else { // Negative Predicate
			int negPredIdx = currentAct- posPredNum;
			StandardPredicate p = allNegPreds[negPredIdx];
			for (int r=0; r<posPredNum; r++) {
				if (allPosPreds[r] == p) {
					predIdx = r;
					break;
				}
			}
		}
		return predIdx;
	}
	
	public int getNegPredIndex(int action) {
		int predIdx = -1;
		if (action == ReturnAction)
			predIdx = -1;
		else if (action >= posPredNum) { // Negative Predicate
			predIdx = action- posPredNum;
		} else { // Positive Predicate
			StandardPredicate p = allPosPreds[action];
			for (int i=0; i<negPredNum; i++) {
				if (allNegPreds[i] == p) {
					predIdx = i;
				}
			}
		}
		return predIdx;
	}
	
	public Integer nextAction() {
		INDArray observation = processHistory();
		INDArray dqnOutput = currentDQN.output(observation);
		int act = Nd4j.argMax(dqnOutput, 1).getInt(0);
		return act;
	}
 	
	public boolean checkIsNetworkRule(int[] lastRulePreds) {	
		Set<StandardPredicate> friendPredSet = new HashSet<StandardPredicate>(Arrays.asList(friendPreds));
		for (int i=0; i<posPredNum; i++) {
			if (lastRulePreds[i]!=0) {
				if (friendPredSet.contains(allPosPreds[i]))
					return true;
			}
		}
		return false;
	}
	
	public boolean checkValideNetworkRule(int[] rule) {
		Set<StandardPredicate> networkPredSet = new HashSet<StandardPredicate>(Arrays.asList(networkPreds));
		for (int i=0; i<posPredNum; i++) {
			if (rule[i] ==1) {
				if (networkPredSet.contains(allPosPreds[i]))
					return true;
			}
		}
		return false;
	}
	
	public boolean checkContainsNegTargetInSeq() {
		int posTargetSignal = 0;
		int targetIdx = posPredNum+ negPredNum-1;
		for (int i=0; i<inputRow; i++) {
			if (ruleListEmbedding[i][targetIdx]==1) {
				posTargetSignal++;
			}
		}
		return posTargetSignal>0 ? true:false;
	}
	
	public INDArray processHistory() {
		INDArray observation;
		observation = Nd4j.zeros(new int[] {1, inputRow*inputCol});
		for (int r=0; r<inputRow; r++) {
			if (IntStream.of(ruleListEmbedding[r]).sum() == 0)
				break;
			for (int c=0; c<inputCol; c++) {
				if (ruleListEmbedding[r][c]==1)
					observation.putScalar(new int[] {0, r*inputCol+c}, 1.0);
			}
		}
		return observation;
	}
	
	protected Pair<INDArray, INDArray> setTarget(ArrayList<StateActTrans> transitions) {
		if (transitions.size()==0) 
			throw new IllegalArgumentException("too few transitions");
		int size = transitions.size();
		
		int[] shape = new int[] {inputRow*inputCol};
		int[] nshape = makeShape(size, shape);
		INDArray obs = Nd4j.create(nshape);
		INDArray nextObs = Nd4j.create(nshape);
		
		int[] actions = new int[size];
		boolean[] areTerminal = new boolean[size];
		for (int i=0; i<size; i++) {
			StateActTrans trans = transitions.get(i);
			actions[i] = trans.action;
			areTerminal[i] = trans.isTerminal;
			
			INDArray obsArray = trans.observation;
			obs.putRow(i, obsArray);
			
			INDArray nextObsArray = trans.nextObservation;
			nextObs.putRow(i, nextObsArray);
		}
		
		INDArray dqnOutput = currentDQN.output(obs);
		INDArray targetDqnOutputNext = targetDQN.output(nextObs); //targetDQN.output(nextObs);
		INDArray getMaxAction = Nd4j.argMax(targetDqnOutputNext, 1);
		
		for (int i=0; i<size; i++) {
			double yTar = transitions.get(i).getReward();
			if (!areTerminal[i]) {
				yTar += GAMMA* targetDqnOutputNext.getDouble(i, getMaxAction.getInt(i));
			}
			dqnOutput.putScalar(i, actions[i], yTar);
		}
		return new Pair<INDArray, INDArray>(obs, dqnOutput);
	}
	
	public double AUCcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		// Define Partitions and Databases
		Partition targetPart = new Partition(2);
		Partition inferenceWritePart = new Partition(3);
		
		Set<StandardPredicate> closedPredicates = new HashSet<StandardPredicate>(Arrays.asList(X));
		Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
		Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
		
		ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(Y[0]));
		for (int i=0; i<allGroundings.size(); i++) {
			GroundTerm [] grounding = allGroundings.get(i);
			GroundAtom inferAtom = inferenceDB.getAtom(Y[0], grounding);
			GroundAtom wlAtom = wlTrainDB.getAtom(Y[0], grounding);
			if (inferAtom instanceof RandomVariableAtom) {
				inferenceDB.commit((RandomVariableAtom) inferAtom);
			}
			if (wlAtom instanceof RandomVariableAtom) {
				wlTrainDB.commit((RandomVariableAtom) wlAtom);
			}
		}
		
		// Do weight Learning
		WeightLearningApplication weightLearner = null;
		if (Z.length>0) {
			weightLearner = new HardEM(model, wlTrainDB, wlTruthDB, config);
		} else {
			weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
		}
	    ArrayList<Integer> numGroundedList = weightLearner.learn();
	    weightLearner.close();
	    wlTrainDB.close();
	    data.deletePartition(targetPart);
	    
	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();

		// Do Inference
		MPEInference mpe = new MPEInference(model, inferenceDB, config);
		FullInferenceResult result = mpe.mpeInference();
		inferenceDB.close();
		
		Set<StandardPredicate> InferredSet = new HashSet<StandardPredicate>(Arrays.asList(Y)); 
		Database resultDB = data.getDatabase(inferenceWritePart, InferredSet);
		SimpleRankingComparator comparator = new SimpleRankingComparator(resultDB);
		comparator.setBaseline(wlTruthDB);
		double[] score = new double[metrics.length];
		for (int r=0; r<metrics.length; r++) {
			comparator.setRankingScore(metrics[r]);
			score[r] = comparator.compare(Y[0]);
		}
		resultDB.close();
		data.deletePartition(inferenceWritePart);

		double reward_auc = 0;
		for (int i=0; i<LAMBDA_AUC.length; i++) {
			reward_auc += score[i]* LAMBDA_AUC[i];
		}
		if (Double.isNaN(reward_auc)) {
			reward_auc = MIN_VAL;
		}
		double reward = reward_auc; //+ reward_grounded;
		if (loss_grounded == 0) 
			reward = 0;
		return reward;
		
	}
	
	
//	public double interpretabilityCost() {
//		double loss_len = 0;
//		double loss_num = 0;
//		int[][] ruleListCopy = new int[inputRow][inputCol];
//		IntStream.range(0, inputRow).forEach(r-> 
//			IntStream.range(0, inputCol).forEach(c-> ruleListCopy[r][c]=ruleListEmbedding[r][c]));
//		// First Remove all the "Return" Actions
//		/*
//		 * Rule Body Length
//		 * Rule List Size
//		 * Coverage
//		 * Overlap
//		 * Diversity
//		 */
//		int listSize = 0;
//		double sumLen = 0;
//		ArrayList<Integer> ruleLenList = new ArrayList<Integer>();
//		for (int i=0; i<ruleListCopy.length; i++) { 
//			if (ruleListCopy[i][BodyReturnAction]==1) // Return action in Body
//				ruleListCopy[i][BodyReturnAction] = 0;
//			if (ruleListCopy[i][HeadReturnAction]==1) // Return action in Head
//				ruleListCopy[i][HeadReturnAction] = 0;
//			
//			int ruleLen = IntStream.of(ruleListCopy[i]).sum();
//			if (ruleLen > 0) {
//				listSize++;
//				ruleLenList.add(ruleLen);
//				sumLen += ruleLen;
//			} else {
//				break;
//			}
//		}
//		loss_len = -sumLen*1.0/listSize;
//		loss_num = -listSize;
////		reward += LAMBDA_ruleLen* 2* (1- 1.0/(1+Math.exp(-loss_len)));
////		reward += LAMBDA_ruleNum* 2* (1- 1.0/(1+Math.exp(-loss_num)));
//		
//		double reward = LAMBDA_ruleLen* loss_len; 
//		reward += LAMBDA_ruleNum* loss_num;
//				
//		return reward;
//	}
	
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		final double initWeight = 5.0;
		
		// Randomly Choose Head Predicate
		List<Integer> potentialHeadPreds = new ArrayList<Integer>();
		for (int i=0; i<ReturnAction; i++) {
//			int negActIdx = i+posPredNum;
//			int posActIdx = getPosPredIndex(negActIdx);
			if (ruleEmbedding[i] == 1) {
				potentialHeadPreds.add(i);
			}
		}
		int headIdx = potentialHeadPreds.get(rdm.nextInt(potentialHeadPreds.size()));
		
//		// Choose Target as Head Predicate
//		int posActIdx = posPredNum-1;
//		int negActIdx = posPredNum+negPredNum-1;
//		int headIdx;
//		if (ruleEmbedding[posActIdx] == 1)
//			headIdx = posActIdx;
//		else
//			headIdx = negActIdx;
//		
		if (headIdx < posPredNum) { // Negative Head
			StandardPredicate p = allPosPreds[headIdx];
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			head = (FormulaContainer) head.bitwiseNegate();
		} else { // Positive Head
			StandardPredicate p = allNegPreds[headIdx-posPredNum];
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
//			head = (FormulaContainer) head.bitwiseNegate();
		}
		
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1 && i!=headIdx) {
				if (i<posPredNum) { // Positive Body Predicate
					StandardPredicate p = allPosPreds[i];
					List<GenericVariable> argsList= generalPredArgsMap.get(p);
					Object[] args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null)
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Negative Body Predicate
					StandardPredicate p = allNegPreds[i-posPredNum];
					List<GenericVariable> argsList= generalPredArgsMap.get(p);
					Object[] args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null) {
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.bitwiseNegate();
					} else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						body = (FormulaContainer) body.and(f_tmp);
					}
				}
			}
		}
		
		if (body==null) {
			rule = head;
		} else {
			rule = (FormulaContainer) body.rightShift(head);
		}
		
		boolean succeed;
		Map<String, Object> argsMap = new HashMap<String, Object>();
		argsMap.put("rule", rule);
		argsMap.put("squared", true);
		argsMap.put("weight", initWeight);
		try {
			model.add(argsMap);
			succeed = true;
		} catch (Exception e) {
			Log.error(e);
			succeed = false;
		}
		
		return succeed;
	}
	
}



