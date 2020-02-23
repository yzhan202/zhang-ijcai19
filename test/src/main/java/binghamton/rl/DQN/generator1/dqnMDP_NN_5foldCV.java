package binghamton.rl.DQN.generator1;

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
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;


public class dqnMDP_NN_5foldCV {
	
	final int offset; //num step for eps greedy anneal
	final double maxAlpha; 
	final double minAlpha; //min epsilon
	
	final StandardPredicate[] X;
	final StandardPredicate[] Y;
	final StandardPredicate[] Z;
	final StandardPredicate[] negFeatPreds;
	final StandardPredicate[] allPosPreds;
	final StandardPredicate[] allNegPreds;
	
	final StandardPredicate[] friendPreds;
	final StandardPredicate[] networkPreds;
	final StandardPredicate[] friendInfoPreds;
	final StandardPredicate[] networkHeadPreds;
	
	final Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	final Map<StandardPredicate, List<List<Object>>> specificPredArgsMap;
	final int maxRuleBodyLen;
	final int maxRuleNum;
	final int posPredNum;
	final int negPredNum;
	
	PSLModel model;
	DataStore data;
	Database wlTrainDB;
	Database wlTruthDB;
	Database inferenceDB;
	Partition trainPart;
	Partition testLabelPartition;
	Partition inferenceWritePart;
	
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
	Random rdm = new Random();
    final int NEURAL_NET_ITERATION_LISTENER = 10000;
    final int updateStart = 10;
    
	ExperienceReplay expReplay; 
	
	final Map<String,Double> groundTruthMap; 
	Map<String, Double> predicationMap;
	
//	int STOP_SIGNAL;
	
	final int inputRow;
	final int inputCol;
	final int outputSize;
	
	final int BodyReturnAction; //Rule Body Return
	final int HeadReturnAction; // Rule Head Return
	
	final double LAMBDA_ruleLen = 0.0;//0.5;
	final double LAMBDA_ruleNum = 0.0;//0.5;
	final double LAMBDA_coverage = 0.0;//0.2;
	
//	final double redundantPenalty = -1;
//	final double targetPenalty = -1; //-1.0;
//	final double grammarPenalty = -1; //-1.0;
//	final double emptyPenalty = -20; //-20.0; //-Double.MAX_VALUE;
	
//	final double NoRedundantReward = 0.01;
//	final double TargetReward = 0.2;
//	final double GrammarReward = 0.2;
//	final double NotEmptyReward = 1.0;
	
	final double REWARD_THRESHOLD;
	final double MIN_VAL = 1e-6;
		
	final double Scaling = 1.0; //20.0;
	
	private ConfigBundle config;
	final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
	final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};

	final int lastEpoch;
	
	public dqnMDP_NN_5foldCV(StandardPredicate[] X, StandardPredicate[] Y, StandardPredicate[] Z, StandardPredicate[] negFeatPreds,
			PSLModel model, DataStore data, Database wlTrainDB, Database wlTruthDB, Database inferenceDB, 
			Partition trainPart, Partition testLabelPartition, ConfigBundle config, int lastEpoch,
			Map<StandardPredicate,List<GenericVariable>> generalPredArgsMap, Map<StandardPredicate,List<List<Object>>> specificPredArgsMap,
			StandardPredicate[] friendPreds, StandardPredicate[] networkPreds, StandardPredicate[] friendInfoPreds, StandardPredicate[] networkHeadPreds) {
		this.maxRuleBodyLen = 5; 
		this.maxRuleNum = 8;
		
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		this.negFeatPreds = negFeatPreds;
		this.posPredNum = X.length+ Y.length+ Z.length;
		this.negPredNum = negFeatPreds.length+ 1;
		this.allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(this.X, this.Y), this.Z);
		this.allNegPreds = ArrayUtils.addAll(negFeatPreds, this.Y);
		this.generalPredArgsMap = generalPredArgsMap;
		this.specificPredArgsMap = specificPredArgsMap;
		this.friendPreds = friendPreds;
		this.networkPreds = networkPreds;
		this.friendInfoPreds = friendInfoPreds;
		this.networkHeadPreds = networkHeadPreds;
		
		this.model = model;
		this.data = data;
		this.wlTrainDB = wlTrainDB;
		this.wlTruthDB = wlTruthDB;
		this.trainPart = trainPart;
		this.testLabelPartition = testLabelPartition;
		
		this.inferenceDB = inferenceDB;
		
		this.outputSize = posPredNum+ negPredNum+ 1; // Positive and Negative Predicates and "Return"
		this.inputRow = maxRuleNum;
		this.inputCol = 2* outputSize; // Body, Head
 		
		this.ruleListEmbedding = new int[inputRow][inputCol];
		
		this.maxEpisode = (int)1e+4;
		this.batchSize = 32;
		
		this.offset = 3000; //(maxRuleBodyLen+1)*maxRuleNum* 200; //(int)2e+4;
		this.maxAlpha = 1.0;
		this.minAlpha = 0.1;
		
		//this.maxStep = maxRuleBodyLen* maxRuleNum;
		currentDQN = buildDenseDQN(new int[] {inputRow, inputCol}, outputSize);
		targetDQN = currentDQN.clone();
		
		GAMMA = 1.0; //0.99;
		targetDqnUpdateFreq = 500;
		expRepMaxSize = 500;
		
		expReplay = new ExperienceReplay(expRepMaxSize, batchSize, NEURAL_NET_SEED);
				
		groundTruthMap = getGroundTruth();
		predicationMap = new HashMap<String, Double>();
				
		BodyReturnAction = outputSize-1; //Rule Body Return
		HeadReturnAction = inputCol-1; // Rule Head Return
		
		REWARD_THRESHOLD = calculateThreshold();
		
		this.config = config;
		this.lastEpoch = lastEpoch;
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
	
    public Map<String, Double> getGroundTruth() {
    	Map<String, Double> groundTruthMap = new HashMap<String, Double>();
    	for (GroundAtom atom : Queries.getAllAtoms(wlTruthDB, Y[0])) {
    		GroundTerm[] arguments = atom.getArguments();
    		String user = arguments[0].toString();
    		groundTruthMap.put(user, atom.getValue());
    	}
    	return groundTruthMap;
    }
    
    public Map<String, Double> getPredication() {
    	return predicationMap;
    }
    
    
    public double calculateThreshold() {
    	double threshold = 0;
    	// BCE
    	int userNum = groundTruthMap.size();
    	threshold += Math.log(MIN_VAL)* userNum;
    	// Rule Length
    	threshold -= LAMBDA_ruleLen* (maxRuleBodyLen+1);
    	// Rule Number
    	threshold -= LAMBDA_ruleNum* maxRuleNum;
    	// Coverage
    	threshold += LAMBDA_coverage* 0;
    	return -threshold;
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
        final int numHiddenNodes = 16;
        final double l2 = 0.0001;
        final double adam = 0.01; //0.001
       
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
		int[] lastBodyPreds = new int[posPredNum];
		int currentAct;
		double reward;
		double accumulatedReward=0;
		
		boolean END_SIGNAL;
		boolean RULE_END_SIGNAL;
		boolean BODY_END_SIGNAL;
			
		boolean REDUNDANT_PENALTY_SIGNAL;
		boolean TARGET_PENALTY_SIGNAL;
		boolean GRAMMAR_PENALTY_SIGNAL;
//		boolean EMPTY_PENALTY_SIGNAL;
		boolean NETWORK_PENALTY_SIGNAL;
		
		/*
		 *  Initialization
		 */
		cleanPSLrule();
		resetRuleListEmbedding();
		nextAccessIdx = 0;
		IntStream.range(0, posPredNum).forEach(r->lastBodyPreds[r]=0);
		END_SIGNAL = false;
		RULE_END_SIGNAL = false;
		BODY_END_SIGNAL = false;
//		currentAct = 0;
		accumulatedReward = 0;
		
		REDUNDANT_PENALTY_SIGNAL = false; // Check on each Action
		TARGET_PENALTY_SIGNAL = false; // Check on each Rule
		GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
//		EMPTY_PENALTY_SIGNAL = false; // Check on Rule List
		NETWORK_PENALTY_SIGNAL = false; // Check on each Rule
		
		int countStep = 0;
		while(true) {
			REDUNDANT_PENALTY_SIGNAL = false;
			
			/*
			 * Greedy Action
			 */			
			currentAct = nextAction();
			
			INDArray observation = processHistory();
			INDArray dqnOutput = currentDQN.output(observation).reshape(outputSize);
			System.out.println(""+ countStep+ ": "+ currentAct+ ", "+ dqnOutput);
			countStep++;
			
			/*
			 * State-Action Transition
			 */
			if (!BODY_END_SIGNAL) { // Choose Body Predicate						
				if (currentAct == BodyReturnAction) { // Rule Body Return
					BODY_END_SIGNAL = true;
					ruleListEmbedding[nextAccessIdx][BodyReturnAction] = 1;
				} else {
					int predIdx = getPosPredIndex(currentAct);
					lastBodyPreds[predIdx] += 1;
					if (IntStream.of(lastBodyPreds).sum() >= maxRuleBodyLen) {
						BODY_END_SIGNAL = true;
					}
					if (lastBodyPreds[predIdx]==1)
						ruleListEmbedding[nextAccessIdx][currentAct] = 1;
				}
			} else if (!RULE_END_SIGNAL){ // Choose Head Predicate
				if (currentAct==BodyReturnAction) { // Rule Head Return
					// Check Grammar
					if (IntStream.of(lastBodyPreds).sum()>0) {
						GRAMMAR_PENALTY_SIGNAL = true;
					}
					ruleListEmbedding[nextAccessIdx][HeadReturnAction] = 1;
				} else {
					int posPredIdx = getPosPredIndex(currentAct);
					int negPredIdx = getNegPredIndex(currentAct);
//					if (lastBodyPreds[predIdx]!=0) { // Check Redundant Predicate
//						REDUNDANT_PENALTY_SIGNAL = true;
//					} else {
					ruleListEmbedding[nextAccessIdx][posPredIdx] = 0;
					if (negPredIdx != -1)
						ruleListEmbedding[nextAccessIdx][negPredIdx+ posPredNum] = 0;
					ruleListEmbedding[nextAccessIdx][currentAct+ outputSize] = 1;
					if (posPredIdx <X.length) { // Check if the rule contains target predicate
						boolean searchTargetPredicate = false;
						for (int t=X.length; t<posPredNum; t++) {
							if (lastBodyPreds[t]!=0) {
								searchTargetPredicate = true;
								break;
							}
						}
						if(!searchTargetPredicate) {
							TARGET_PENALTY_SIGNAL = true;
						}
					}
//					}
					// Check Network Rule
					if (checkIsNetworkRule(ruleListEmbedding[nextAccessIdx])) {
						Set<StandardPredicate> networkHeadPredSet = new HashSet<StandardPredicate>(Arrays.asList(networkHeadPreds));
						NETWORK_PENALTY_SIGNAL = !checkValideNetworkRule(ruleListEmbedding[nextAccessIdx]) 
								|| !checkFriendInfoInValideNetworkRule(ruleListEmbedding[nextAccessIdx]) || !(networkHeadPredSet.contains(allPosPreds[posPredIdx]));
					}
				}
				// Check Grammar
				if ((!REDUNDANT_PENALTY_SIGNAL) && (!TARGET_PENALTY_SIGNAL) && (!NETWORK_PENALTY_SIGNAL)
						&& (currentAct!=BodyReturnAction)) {
					boolean succeed = buildNewRule(ruleListEmbedding[nextAccessIdx]);
					if (!succeed) {
						GRAMMAR_PENALTY_SIGNAL = true;
					}
				}
				RULE_END_SIGNAL = true;
			}
			
			/*
			 * Assign Reward
			 */
			reward = 0;
			if (REDUNDANT_PENALTY_SIGNAL || TARGET_PENALTY_SIGNAL || GRAMMAR_PENALTY_SIGNAL) {
				END_SIGNAL = true;
			}
			if (RULE_END_SIGNAL) {
				nextAccessIdx++;
				// Check Finishing Building Rule List
				if (!END_SIGNAL)
					if ((nextAccessIdx == maxRuleNum) || (currentAct==BodyReturnAction)) {
						END_SIGNAL = true;
					}
				IntStream.range(0, posPredNum).forEach(r->lastBodyPreds[r]=0);
			}
			if (END_SIGNAL) {
				Iterator<Kernel> kernels = model.getKernels().iterator();
				int kernelSize = 0;
				for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
					kernelSize++;
				}
				if (kernelSize!=0) {
					// Clean Inference Database
					cleanInferenceDatabase();
					double lossFunction_reward = 0; //REWARD_THRESHOLD;					
//					// Reward from Binary Classification Entropy
//					double tmp = BCEcost();
//					lossFunction_reward += tmp;
					double tmp1 = AUCcost();
					lossFunction_reward += tmp1;
					//Reward from Interpretability Constaints
					double tmp2 = interpretabilityCost();
					lossFunction_reward += tmp2;
					
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
				BODY_END_SIGNAL = false;
				RULE_END_SIGNAL = false;
				TARGET_PENALTY_SIGNAL = false;
				GRAMMAR_PENALTY_SIGNAL = false;
				REDUNDANT_PENALTY_SIGNAL = false;
				NETWORK_PENALTY_SIGNAL = false;
			}
			if (END_SIGNAL) {
				break;
			}
		}
		SimpleRankingComparator comparator = new SimpleRankingComparator(inferenceDB);
		comparator.setBaseline(wlTruthDB);
		double[] score = new double[metrics.length];
		for (int r=0; r<metrics.length; r++) {
			comparator.setRankingScore(metrics[r]);
			score[r] = comparator.compare(Y[0]);
		}
		System.out.println(model.toString());
		System.out.println("Area under positive-class PR curve: "+ score[0]);
		System.out.println("Area under negative-class PR curve: "+ score[1]);
		System.out.println("Area under ROC curve: "+ score[2]);
		return accumulatedReward;
	}
	
	
	public double[] training() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		int nextAccessIdx;
		int[] lastBodyPreds = new int[posPredNum];
		int currentAct;
		int stepCounter = 0;
		double alpha=1.0;
		double reward;
		int kernelSize = 0;
		
		double accumulatedReward;
		
		boolean END_SIGNAL;
		boolean RULE_END_SIGNAL;
		boolean BODY_END_SIGNAL;
				
		boolean REDUNDANT_PENALTY_SIGNAL;
		boolean TARGET_PENALTY_SIGNAL;
		boolean GRAMMAR_PENALTY_SIGNAL;
//		boolean EMPTY_PENALTY_SIGNAL;
		boolean NETWORK_PENALTY_SIGNAL;
		
		double[] trend = new double[maxEpisode];
		for (int epoch=0; epoch<maxEpisode; epoch++) {
			/*
			 *  Initialization
			 */
			cleanPSLrule();
			resetRuleListEmbedding();
			nextAccessIdx = 0;
			IntStream.range(0, posPredNum).forEach(r->lastBodyPreds[r]=0);
			END_SIGNAL = false;
			RULE_END_SIGNAL = false;
			BODY_END_SIGNAL = false;
//			currentAct = 0;
			accumulatedReward = 0;
			
			REDUNDANT_PENALTY_SIGNAL = false; // Check on each Action
			TARGET_PENALTY_SIGNAL = false; // Check on each Rule
			GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
			NETWORK_PENALTY_SIGNAL = false; // Check on each RUle
			
			alpha = epsilonSchedule(epoch+ lastEpoch);
			while(!END_SIGNAL) {
//				alpha = epsilonSchedule(stepCounter);
				REDUNDANT_PENALTY_SIGNAL = false;
				
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
				if (!BODY_END_SIGNAL) { // Choose Body Predicate						
					if (currentAct == BodyReturnAction) { // Rule Body Return
						BODY_END_SIGNAL = true;
						ruleListEmbedding[nextAccessIdx][BodyReturnAction] = 1;
					} else {
						int predIdx = getPosPredIndex(currentAct);
						lastBodyPreds[predIdx] += 1;
						if (IntStream.of(lastBodyPreds).sum() >= maxRuleBodyLen) {
							BODY_END_SIGNAL = true;
						}
						if (lastBodyPreds[predIdx]==1)
							ruleListEmbedding[nextAccessIdx][currentAct] = 1;
					}
				} else if (!RULE_END_SIGNAL){ // Choose Head Predicate
					if (currentAct==BodyReturnAction) { // Rule Head Return
						// Check Grammar
						if (IntStream.of(lastBodyPreds).sum()>0) {
							GRAMMAR_PENALTY_SIGNAL = true;
						}
						ruleListEmbedding[nextAccessIdx][HeadReturnAction] = 1;
					} else {
						int posPredIdx = getPosPredIndex(currentAct);
						int negPredIdx = getNegPredIndex(currentAct);
//						if (lastBodyPreds[predIdx]!=0) { // Check Redundant Predicate
//							REDUNDANT_PENALTY_SIGNAL = true;
//						} else {
						ruleListEmbedding[nextAccessIdx][posPredIdx] = 0;
						if (negPredIdx != -1)
							ruleListEmbedding[nextAccessIdx][negPredIdx+ posPredNum] = 0;
						ruleListEmbedding[nextAccessIdx][currentAct+ outputSize] = 1;
						if (posPredIdx <X.length) { // Check if the rule contains target predicate
							boolean searchTargetPredicate = false;
							for (int t=X.length; t<posPredNum; t++) {
								if (lastBodyPreds[t]!=0) {
									searchTargetPredicate = true;
									break;
								}
							}
							if(!searchTargetPredicate) {
								TARGET_PENALTY_SIGNAL = true;
							}
						}
//						}
						// Check Network Rule
						if (checkIsNetworkRule(ruleListEmbedding[nextAccessIdx])) {
							Set<StandardPredicate> networkHeadPredSet = new HashSet<StandardPredicate>(Arrays.asList(networkHeadPreds));
							NETWORK_PENALTY_SIGNAL = !checkValideNetworkRule(ruleListEmbedding[nextAccessIdx]) 
									|| !checkFriendInfoInValideNetworkRule(ruleListEmbedding[nextAccessIdx]) 
									|| !(networkHeadPredSet.contains(allPosPreds[posPredIdx]));
						}
					}
					// Check Grammar
					if ((!REDUNDANT_PENALTY_SIGNAL) && (!TARGET_PENALTY_SIGNAL) && (!NETWORK_PENALTY_SIGNAL)
							&& (currentAct!=BodyReturnAction)) {
						boolean succeed = buildNewRule(ruleListEmbedding[nextAccessIdx]);
						if (!succeed) {
							GRAMMAR_PENALTY_SIGNAL = true;
						}
					}
					RULE_END_SIGNAL = true;
				}
				
				/*
				 * Assign Reward
				 */
				reward = 0;
				if (REDUNDANT_PENALTY_SIGNAL || TARGET_PENALTY_SIGNAL
						|| GRAMMAR_PENALTY_SIGNAL || NETWORK_PENALTY_SIGNAL) {
					END_SIGNAL = true;
				}
				if (RULE_END_SIGNAL) {
					nextAccessIdx++;
					// Check Finishing Building Rule List
					if (!END_SIGNAL)
						if ((nextAccessIdx == maxRuleNum) || (currentAct==BodyReturnAction)) {
							END_SIGNAL = true;
						}
					IntStream.range(0, posPredNum).forEach(r->lastBodyPreds[r]=0);
				}
				if (END_SIGNAL) {
					Iterator<Kernel> kernels = model.getKernels().iterator();
					kernelSize = 0;
					for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)){
						kernelSize++;
					}
					if (kernelSize!=0) {
						// Clean Inference Database
						cleanInferenceDatabase();
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
				
				StateActTrans transition = new StateActTrans(observation, currentAct, (!RULE_END_SIGNAL), reward, END_SIGNAL, nextObservation); 
				expReplay.store(transition);
				
				/*
				 * Sample Random minibatch
				 */
				if (stepCounter > updateStart) {
					ArrayList<StateActTrans> transBatch = expReplay.getBatch();
					Pair<INDArray, INDArray> target_data = setTarget(transBatch);	
					currentDQN.fit(target_data.getFirst(), target_data.getSecond());
				}
				
				stepCounter++;
				if (stepCounter % targetDqnUpdateFreq == 0) {
					updateTargetNetwork();
				}
				
				/*
				 * Reset Signals
				 */
				if (RULE_END_SIGNAL) {
					BODY_END_SIGNAL = false;
					RULE_END_SIGNAL = false;
					
					TARGET_PENALTY_SIGNAL = false;
					GRAMMAR_PENALTY_SIGNAL = false;
					REDUNDANT_PENALTY_SIGNAL = false;
					NETWORK_PENALTY_SIGNAL = false;
				}
			}
			
			trend[epoch] = accumulatedReward;
			if (epoch % 50 ==0) {
				System.out.println("Epoch: "+ epoch+ ", Step Counter: "+ stepCounter+ 
						", Alpha: "+ alpha+ ", Cummulated Reward: "+ accumulatedReward+ ", Kernel Size: "+ kernelSize);
				if (kernelSize > 0) {
					System.out.println(model.toString());
//					if (accumulatedReward > 0.0) {
//						SimpleRankingComparator comparator = new SimpleRankingComparator(inferenceDB);
//						comparator.setBaseline(wlTruthDB);
//						double[] score = new double[metrics.length];
//						for (int r=0; r<metrics.length; r++) {
//							comparator.setRankingScore(metrics[r]);
//							score[r] = comparator.compare(Y[0]);
//						}
//						System.out.println("Area under positive-class PR curve: "+ score[0]);
//						System.out.println("Area under negative-class PR curve: "+ score[1]);
//						System.out.println("Area under ROC curve: "+ score[2]);
//					}
				}
			}
		}
		
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
	
	
	public void cleanInferenceDatabase() {
//		inferenceDB.close();
		if (inferenceWritePart != null)
			data.deletePartition(inferenceWritePart);
		
        StandardPredicate predRecovers = Y[0]; //(StandardPredicate)PredicateFactory.getFactory().getPredicate("Recovers");
		inferenceWritePart = new Partition(321);	
		Set<StandardPredicate> setX = new HashSet<StandardPredicate>(Arrays.asList(X));
		inferenceDB = data.getDatabase(inferenceWritePart, setX, new Partition[] {trainPart, testLabelPartition});
	
		ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(predRecovers));
		for (int i=0; i<allGroundings.size(); i++) {
			GroundTerm [] grounding = allGroundings.get(i);
			GroundAtom atom = inferenceDB.getAtom(predRecovers, grounding);
			if (atom instanceof RandomVariableAtom) {
				inferenceDB.commit((RandomVariableAtom) atom);
			}
		}
	}
	
	public int getPosPredIndex(int currentAct) {
		if (currentAct==BodyReturnAction || currentAct==HeadReturnAction)
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
		if (action == BodyReturnAction)
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
 	
	public boolean checkIsNetworkRule(int[] rule) {	
		Set<StandardPredicate> friendPredSet = new HashSet<StandardPredicate>(Arrays.asList(friendPreds));
		for (int i=0; i<inputCol; i++) {
			if (rule[i] != 0) {
				if (i < outputSize) { // Body
					if (i!=BodyReturnAction) {
						if (i<posPredNum) { // Positive Predicate
							if (friendPredSet.contains(allPosPreds[i])) 
								return true;
						} else { // Negative Predicate
							if (friendPredSet.contains(allNegPreds[i-posPredNum]))
								return true;
						}
					}
				} else { // Head
					if (i!=HeadReturnAction) {
						if (i<(posPredNum+ outputSize)) { // Positive Predicate
							if (friendPredSet.contains(allPosPreds[i- outputSize]))
								return true;
						} else { // Negative Predicate
							if (friendPredSet.contains(allNegPreds[i- outputSize- posPredNum]))
								return true;
						}
					}
				}
			}
		}
		return false;
	}
	
	public boolean checkValideNetworkRule(int[] rule) {
		Set<StandardPredicate> networkPredSet = new HashSet<StandardPredicate>(Arrays.asList(networkPreds));
		for (int i=0; i<inputCol; i++) {
			if (rule[i] != 0) {
				if (i < outputSize) { // Body
					if (i!=BodyReturnAction) {
						if (i < posPredNum) { // Positive Predicate
							if (networkPredSet.contains(allPosPreds[i]))
								return true;
						} 
					}
				} 
			}
		}
		return false;
	}
	
	public boolean checkFriendInfoInValideNetworkRule(int[] rule) {
		Set<StandardPredicate> friendInfoPredSet = new HashSet<StandardPredicate>(Arrays.asList(friendInfoPreds));
		for (int i=0; i<inputCol; i++) {
			if (rule[i] != 0) {
				if (i < outputSize) { // Body
					if (i!=BodyReturnAction) {
						if (i < posPredNum) { // Positive Predicate
							if (friendInfoPredSet.contains(allPosPreds[i]))
								return true;
						} else { // Negative Predicate
							if (friendInfoPredSet.contains(allNegPreds[i-posPredNum]))
								return true;
						}
					}
				} else { // Head
					if (i!=HeadReturnAction) {
						if (i < (outputSize+ posPredNum)) { // Positive Predicate
							if (friendInfoPredSet.contains(allPosPreds[i- outputSize]))
								return true;
						} else { // Negative Predicate
							if (friendInfoPredSet.contains(allNegPreds[i- (outputSize+posPredNum)]))
								return true;
						}
					}
				}
			}
		}
		return false;
	}
	
	public INDArray processHistory() {
//		System.out.println("Process Observation");
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
		INDArray targetDqnOutputNext = targetDQN.output(nextObs);
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
	
	
	public double BCEcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		double reward_grounded = weightLearning();	
		
		runInference();
		// Get Predication Value
		predicationMap.clear();
		for (GroundAtom atom : Queries.getAllAtoms(inferenceDB, Y[0])) {
			GroundTerm[] terms = atom.getArguments();
			String user = terms[0].toString();
			double predicatedValue = atom.getValue();
			predicationMap.put(user, predicatedValue);
		}
		
		/*
		 * Calculate Binary Classification Entropy
		 */
		double reward_bce = 0;
		for ( String user : predicationMap.keySet() ) {
			double groundTruth = groundTruthMap.get(user);
			double predicatedValue = predicationMap.get(user);
			
			double l;
			if (groundTruth==0.0) {
				l = Math.log(1- predicatedValue+ MIN_VAL);
			} else {
				l = Math.log(predicatedValue+ MIN_VAL);
			}
			reward_bce += l;
		}
		
//		double reward = 2.0/(Math.exp(-loss_BCE));
//		System.out.println("BCE Loss: "+ loss_BCE+ " reward: "+ reward);
		double reward = reward_bce+ reward_grounded;
		
		return reward;
	}
	
	public double AUCcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		// Do weight Learning
		double reward_grounded = weightLearning();
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
//		System.out.println("Area under positive-class PR curve: "+ score[0]);
//		System.out.println("Area under negative-class PR curve: "+ score[1]);
//		System.out.println("Area under ROC curve: "+ score[2]);
		double reward_auc = 0;
		for (int i=0; i<LAMBDA_AUC.length; i++) {
			reward_auc += score[i]* LAMBDA_AUC[i];
		}
		if (Double.isNaN(reward_auc)) {
			reward_auc = MIN_VAL;
		}
//		System.out.println(" After Inference AUC: "+ reward_auc+ model.toString());
		double reward = reward_auc+ reward_grounded;
		return reward;
	}
	
	
	public double interpretabilityCost() {
		double loss_len = 0;
		double loss_num = 0;
		int[][] ruleListCopy = new int[inputRow][inputCol];
		IntStream.range(0, inputRow).forEach(r-> 
			IntStream.range(0, inputCol).forEach(c-> ruleListCopy[r][c]=ruleListEmbedding[r][c]));
		// First Remove all the "Return" Actions
		/*
		 * Rule Body Length
		 * Rule List Size
		 * Coverage
		 * Overlap
		 * Diversity
		 */
		int listSize = 0;
		double sumLen = 0;
		ArrayList<Integer> ruleLenList = new ArrayList<Integer>();
		for (int i=0; i<ruleListCopy.length; i++) { 
			if (ruleListCopy[i][BodyReturnAction]==1) // Return action in Body
				ruleListCopy[i][BodyReturnAction] = 0;
			if (ruleListCopy[i][HeadReturnAction]==1) // Return action in Head
				ruleListCopy[i][HeadReturnAction] = 0;
			
			int ruleLen = IntStream.of(ruleListCopy[i]).sum();
			if (ruleLen > 0) {
				listSize++;
				ruleLenList.add(ruleLen);
				sumLen += ruleLen;
			} else {
				break;
			}
		}
		loss_len = -sumLen*1.0/listSize;
		loss_num = -listSize;
//		reward += LAMBDA_ruleLen* 2* (1- 1.0/(1+Math.exp(-loss_len)));
//		reward += LAMBDA_ruleNum* 2* (1- 1.0/(1+Math.exp(-loss_num)));
		
		double reward = LAMBDA_ruleLen* loss_len; 
		reward += LAMBDA_ruleNum* loss_num;
				
		return reward;
	}
	
	
	public double weightLearning() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		WeightLearningApplication weightLearner = null;
		if (Z.length>0) {
			weightLearner = new HardEM(model, wlTrainDB, wlTruthDB, config);
		} else {
			weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
		}
	    ArrayList<Integer> numGroundedList = weightLearner.learn();
	    weightLearner.close();
	    
	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();
	    double reward = LAMBDA_coverage* loss_grounded;
	    
		return reward;
	}
	
	
	public void runInference() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
//		ADMMReasoner reasoner = new ADMMReasoner(config);
//		PersistedAtomManager atomManager = new PersistedAtomManager(inferenceDB);
//		
//		Grounding.groundAll(model, atomManager, reasoner);
//		reasoner.optimize();
//		
//		int count = 0;
//		for (RandomVariableAtom atom : atomManager.getPersistedRVAtoms()) {	
////			log.info("************ Debug for ADMM Reasoner ****************");
////			log.info(atom.toString()+ "\t"+ atom.getValue());
//			atom.commitToDB();
//			count++;
//		}
//		
//		double incompatibility = GroundKernels.getTotalWeightedIncompatibility(reasoner.getCompatibilityKernels());
//		double infeasibility = GroundKernels.getInfeasibilityNorm(reasoner.getConstraintKernels());
//		int size = reasoner.size();
		
		MPEInference mpe = new MPEInference(model, inferenceDB, config);
		FullInferenceResult result = mpe.mpeInference();
		inferenceDB.close();
	}
	
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		
		final double initWeight = 5.0;
		
//		final int BodyReturnAction = 2*PredNum;
//		final int HeadReturnAction = 4*PredNum+1;
		for (int j=0; j<ruleEmbedding.length; j++) {
			if (ruleEmbedding[j]!=0) {
				if (j < outputSize) { // Body Action
					if (j != BodyReturnAction) { 
						if (j < posPredNum) { // Positive Body Predicate
							StandardPredicate p = allPosPreds[j];
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
							StandardPredicate p = allNegPreds[j-posPredNum];
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
				} else { // Head Action
					if (j != HeadReturnAction) {
						if (j < outputSize+ posPredNum) { // Positive Head Predicate
							StandardPredicate p = allPosPreds[j-outputSize];
							List<GenericVariable> argsList= generalPredArgsMap.get(p);
							Object[] args = new Object[argsList.size()];
							args = argsList.toArray(args);
							head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						} else { // Negative Head Predicate
							StandardPredicate p = allNegPreds[j- (outputSize+posPredNum)];
							List<GenericVariable> argsList= generalPredArgsMap.get(p);
							Object[] args = new Object[argsList.size()];
							args = argsList.toArray(args);
							head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
							head = (FormulaContainer) head.bitwiseNegate();
						}
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
		argsMap.put("square", true);
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



