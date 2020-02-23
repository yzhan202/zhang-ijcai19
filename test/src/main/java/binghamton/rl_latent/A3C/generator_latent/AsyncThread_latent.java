package binghamton.rl_latent.A3C.generator_latent;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.Stack;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.log4j.spi.LoggerFactory;
import org.deeplearning4j.nn.gradient.Gradient;
import org.jfree.util.Log;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.collect.Iterables;

import binghamton.rl.NeuralNet;
import binghamton.rl.A3C.ActorCriticSeparate;
import binghamton.rl.A3C.IActorCritic;
import binghamton.rl.A3C.MiniTrans;
import binghamton.rl_latent.A3C.latentPSLModelCreation;
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication;
import edu.umd.cs.psl.application.learning.weight.em.HardEM;
import edu.umd.cs.psl.application.learning.weight.semantic_em.SemanticHardEM;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.semantic.SemanticMaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.database.loading.Inserter;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.util.database.Queries;


public class AsyncThread_latent<NN extends NeuralNet> extends Thread {
	Random rdm;
	
	private int threadNumber;
	private int stepCounter = 0;
	private int epochCounter = 0;
	final int nstep;
	final double gamma;
	final int updateFreq = 1;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	final StandardPredicate UserDummyPred;
	final StandardPredicate[] allPosPreds;
	final StandardPredicate[] allNegPreds;
	Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	Map<StandardPredicate, List<List<Object>>> specificPredArgsMap;
	/*
	 * Right Reasons
	 */
	final Set<String> Bullying_Signal;
	final Set<String> Nonbullying_Signal;
	
	final int maxRuleLen;
	final int maxRuleNum;
	final int posPredNum;
	final int negPredNum;
	
	final int ReturnAction;
	
	PSLModel model;
	DataStore data;
	Database wlTruthDB;
	Partition trainPart;
	
	int[][] ruleListEmbedding;
	
	final int inputRow;
	final int inputCol;
	final int outputSize;
	
	final double MIN_VAL = 1e-5;
	final double LAMBDA_ruleLen = 0.0;//0.5;
	final double LAMBDA_ruleNum = 0.0;//0.5;
	final double LAMBDA_coverage = 0.0;//0.2;
	final double LAMBDA_diversity = 0.0;
	final double LAMBDA_semantic = 0.0;
	
	private NN current;
	private AsyncGlobal_latent<NN> asyncGlobal;
	
	private ConfigBundle config;
	
	public AsyncThread_latent(AsyncGlobal_latent<NN> asyncGlobal, int threadNum,  int maxRuleLen, int maxRuleNum, int nstep, double gamma) {
		this.threadNumber = threadNum;
		this.asyncGlobal = asyncGlobal;
		this.rdm = new Random();
		
		this.maxRuleLen = maxRuleLen;
		this.maxRuleNum = maxRuleNum;
		this.nstep = nstep;
		this.gamma = gamma;
		
		latentPSLModelCreation generator = new latentPSLModelCreation(threadNumber);
		this.data = generator.getData();
		this.model = generator.getModel();
		this.config = generator.getConfig();
		this.wlTruthDB = generator.getWlTruthDB();
		this.trainPart = generator.getTrainPart();
		
		X = generator.getX();
		Y = generator.getY();
		Z = generator.getZ();
		
		// Dummy Predicates
		UserDummyPred = (StandardPredicate)PredicateFactory.getFactory().getPredicate("questionAnswer");
		X = ArrayUtils.removeElement(X, UserDummyPred);
		
		StandardPredicate[] negFeatPreds = generator.getNegFeatPreds();
		this.allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(X, Y), Z);
		this.allNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, Y), Z);
		this.posPredNum = allPosPreds.length;
		this.negPredNum = allNegPreds.length;
		this.generalPredArgsMap = generator.getGeneralPredArgsMap();
		// Right Reasons
		this.Bullying_Signal = generator.getBullying_Signal();
		this.Nonbullying_Signal = generator.getNonBullying_Signal();
		
		outputSize = posPredNum+ negPredNum+ 1; // Positive, Negative Predicates, And Return
				
		inputRow = maxRuleNum;
		inputCol = outputSize;
		this.ReturnAction = outputSize-1;

		ruleListEmbedding = new int[inputRow][inputCol];
		
		synchronized (asyncGlobal) {
            current = (NN)asyncGlobal.getCurrent().clone();
        }
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
	
	@Override
	public void run() { //throws ClassNotFoundException, IllegalAccessException, InstantiationException, FileNotFoundException {
		try {
			System.out.println("ThreadNum-"+ threadNumber+ " Started!");

			double bestScore = 0;
			int bestEpochIdx = 0;
			String bestRuleList = "";
			
			int nextAccessIdx;
			int[] lastRulePreds = new int[posPredNum];
			int currentAct;
			stepCounter = 0;
			double reward;
			int kernelSize = 0;
			double accumulatedReward;
			
			boolean END_SIGNAL;
			boolean RULE_END_SIGNAL;
						
			boolean TARGET_PENALTY_SIGNAL;
			boolean GRAMMAR_PENALTY_SIGNAL;
			boolean NETWORK_PENALTY_SIGNAL;
			
			int t_start;
						
			/*
			 * Initialization
			 */
			current.reset();
			resetRuleListEmbedding();
			cleanPSLrule();
			nextAccessIdx = 0;
			IntStream.range(0, posPredNum).forEach(r->lastRulePreds[r]=0);
			END_SIGNAL = false;
			RULE_END_SIGNAL = false;
			accumulatedReward = 0;
			
			TARGET_PENALTY_SIGNAL = false; // Check on each Rule
			GRAMMAR_PENALTY_SIGNAL = false; // Check on each Rule
			NETWORK_PENALTY_SIGNAL = false; // Check on each Rule
			
			while(!asyncGlobal.isTrainingComplete()) {
				t_start = stepCounter;
				
				if (epochCounter % updateFreq == 0) { // Update Local neural Nets 
					synchronized (asyncGlobal) {
						current.copy(asyncGlobal.getCurrent());
				    }
				}
				Stack<MiniTrans> rewards = new Stack<MiniTrans>();
				while (!END_SIGNAL && (stepCounter-t_start < nstep)) {
					stepCounter++;
					INDArray observation = processHistory();
					currentAct = nextAction();
					
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
							if (lastRulePreds[posPredIdx] == 1) {
								ruleListEmbedding[nextAccessIdx][currentAct] = 1;
							}
							if (IntStream.of(lastRulePreds).sum() >= maxRuleLen) {
								ruleListEmbedding[nextAccessIdx][ReturnAction] = 1;
								RULE_END_SIGNAL = true;
							}
						}
					}
					if (RULE_END_SIGNAL && IntStream.of(lastRulePreds).sum()>0) {
						// Check Target
						if (lastRulePreds[posPredNum-1-Z.length]==0 && lastRulePreds[posPredNum-1]==0)
							TARGET_PENALTY_SIGNAL = true;
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
						if (kernelSize!=0 && checkContainsNegTargetInSeq()) { //  && checkContainsNegTargetInSeq()
//							// Clean Inference Database
//							cleanInferenceDatabase();
							double lossFunction_reward = 0; //REWARD_THRESHOLD;					
//							// Reward from Binary Classification Entropy
//							double tmp = BCEcost();
//							lossFunction_reward += tmp;
							double tmp1 = AUCcost();
							lossFunction_reward += tmp1;
							//Reward from Interpretability Constaints
//							double tmp2 = interpretabilityCost();
//							lossFunction_reward += tmp2;
							
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
					
					/*
					 * Stack
					 */
					INDArray valueOutput = current.outputAll(observation)[0];
					rewards.add(new MiniTrans(observation, currentAct, valueOutput, reward));
				}
				
				INDArray observation = processHistory();
				if (END_SIGNAL) {
					rewards.add(new MiniTrans(observation, null, null, 0));
				} else {
					INDArray valueOutput = current.outputAll(observation)[0];
					double value = valueOutput.getDouble(0);
					rewards.add(new MiniTrans(observation, null, null, value));
				}
				
				if (current.isRecurrent())
					asyncGlobal.enqueue(calcGradient4RNN((IActorCritic) current, rewards), (stepCounter-t_start));
				else {
					asyncGlobal.enqueue(calcGradient4NN((ActorCriticSeparate) current, rewards), (stepCounter-t_start));
				}
								
				if (END_SIGNAL) {
					if (accumulatedReward > bestScore) {
						bestScore = accumulatedReward;
						bestEpochIdx = epochCounter;
						bestRuleList = model.toString();
						System.out.println("Thread "+ threadNumber+ " Update Best Score: "+ bestScore+ " at Epoch "+ 
								bestEpochIdx+ ", "+bestRuleList);
					}
					
					if (epochCounter % 100 == 0 || accumulatedReward >= 0.7) {
						System.out.println("Thread-"+ threadNumber+ " [Epoch: "+ epochCounter+ ", Step: "+ stepCounter+ "]"+ 
								", Reward: "+ accumulatedReward+ ", Size: "+ kernelSize);
						if (kernelSize>0) {
							System.out.println(model.toString());
						}
					}
					epochCounter++;

					current.reset();
					resetRuleListEmbedding();
					cleanPSLrule();
					nextAccessIdx = 0;
					END_SIGNAL = false;
					accumulatedReward = 0;
				}
			}
//			/*
//			 * Save Convergence
//			 */
//			String outputDir = "/home/yue/Public/Java/structureLearning4PSL/test/result/A3C/";
//			PrintStream ps = new PrintStream(new FileOutputStream(outputDir+ "convergent_"+threadNumber+ ".txt", false));
//			System.setOut(ps);
//			for (int i=0; i<trend.size(); i++) {
//				System.out.println(""+ trend.get(i));
//			}
			
			System.out.println("Thread "+ threadNumber+ " Best Score: "+ bestScore+ 
					" at Epoch "+ bestEpochIdx+ ", "+bestRuleList);
			
		} catch (Exception e) {
			System.out.println("Thread crashed: " + e);
		}
	}

	public void resetRuleListEmbedding() {
		IntStream.range(0,inputRow).forEach(i -> 
			IntStream.range(0, inputCol).forEach(j-> ruleListEmbedding[i][j]=0));
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
	
	public boolean checkContainsNegTargetInSeq() { // Y
		int posTargetSignal = 0;
		int targetIdx = posPredNum+ negPredNum-1- Z.length;
		for (int i=0; i<inputRow; i++) {
			if (ruleListEmbedding[i][targetIdx]==1) {
				posTargetSignal++;
			}
		}
		return posTargetSignal>0 ? true:false;
	}
	
	public boolean checkContainsLatentVariable() { // Z
		int latentSignal = 0;
		for (int i=0; i<inputRow; i++) {
			if (ruleListEmbedding[i][posPredNum-1]==1 || ruleListEmbedding[i][posPredNum+negPredNum-1]==1) {
				latentSignal++;
			}
		}
		return latentSignal>0 ? true:false;
	}
	
	public Integer nextAction() {
		INDArray observation = processHistory();
		INDArray policyOutput = current.outputAll(observation)[1].reshape(new int[] {outputSize});
		float rVal = rdm.nextFloat();

		for (int i=0; i<policyOutput.length(); i++) {
			if (rVal < policyOutput.getFloat(i)) {
//				System.out.println("Epoch: "+epochCounter+", Choose Action: "+ i+ ", Policy Output: "+ policyOutput);
				return i;
			}
			else {
				rVal -= policyOutput.getFloat(i);
			}
		}
//		System.out.println("Output from network is not a probability distribution: " + policyOutput);
        throw new RuntimeException("Output from network is not a probability distribution: " + policyOutput);
	}
 	
	public INDArray processHistory() {
//		System.out.println("Process Observation");
		INDArray observation;
		if (current.isRecurrent()) {
			observation = Nd4j.zeros(makeShape(1, new int[] {inputRow, inputCol}, 1));
			for (int r=0; r<inputRow; r++) {
				if (IntStream.of(ruleListEmbedding[r]).sum() == 0)
					break;
				for (int c=0; c<inputCol; c++) {
					if (ruleListEmbedding[r][c]==1)
						observation.putScalar(new int[] {0, r*inputCol+c, 0}, 1.0);
				}
			}
		} else {
			observation = Nd4j.zeros(new int[] {1, inputRow*inputCol});
			for (int r=0; r<inputRow; r++) {
				if (IntStream.of(ruleListEmbedding[r]).sum() == 0)
					break;
				for (int c=0; c<inputCol; c++) {
					if (ruleListEmbedding[r][c]==1)
						observation.putScalar(new int[] {0, r*inputCol+c}, 1.0);
				}
			}
		}
		return observation;
	}
	
	public Gradient[] calcGradient4RNN(IActorCritic iac, Stack<MiniTrans> rewards) {
		MiniTrans minTrans = rewards.pop();
		int size = rewards.size();
		boolean recurrent = asyncGlobal.getCurrent().isRecurrent();
		
		int[] shape = new int[] {inputRow, inputCol};
		int[] nshape = makeShape(1, shape, size);
		
		INDArray input = Nd4j.create(nshape);
		INDArray targets = Nd4j.create(1, 1, size);
		INDArray logSoftmax = Nd4j.zeros(1, outputSize, size);
		
		double r = minTrans.reward;
//		System.out.println("Last Step reward: "+ r);
		for (int i=size-1; i>=0; i--) {
			minTrans = rewards.pop();			
			r = minTrans.reward+ gamma* r;
            input.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)).assign(minTrans.obs);
            // the critic
            targets.putScalar(i, r);
            
//            if (r!=0.0)
//            	System.out.println("Original: "+ current.outputAll(minTrans.obs)[0].getDouble(0)+ ", Target: "+ r);
            //the actor
            double expectedV = minTrans.valueOutput.getDouble(0);
            double advantage = r- expectedV;
            logSoftmax.putScalar(0, minTrans.action, i, advantage);
		}
		return iac.gradient(input, new INDArray[] {targets, logSoftmax});
	}
	
	public Gradient[] calcGradient4NN(ActorCriticSeparate iac, Stack<MiniTrans> rewards) {
		MiniTrans minTrans = rewards.pop();
		int size = rewards.size();
		
//		int[] shape = new int[] {inputRow, inputCol};
		int[] nshape = new int[] {size, inputRow*inputCol}; //makeShape(size, shape);
		INDArray input = Nd4j.create(nshape);
		INDArray targets = Nd4j.create(size, 1);
		INDArray logSoftmax = Nd4j.zeros(size, outputSize);
		
		double r = minTrans.reward;
		for (int i=size-1; i>=0; i--) {
			minTrans = rewards.pop();
			r = minTrans.reward+ gamma*r;
			input.get(NDArrayIndex.point(i), NDArrayIndex.all()).assign(minTrans.obs);
			
			// the critic
			targets.putScalar(i, r);
			// the actor
			double expectedV = minTrans.valueOutput.getDouble(0);
			double advantage = r- expectedV;
			logSoftmax.putScalar(i, minTrans.action, advantage);
		}

		Gradient[] gradients = iac.gradient(input, new INDArray[] {targets, logSoftmax});
		return gradients;
	}
	
	public double AUCcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
		final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};
		
		// Define Partitions and Databases
		Partition targetPart = new Partition(2);
		Partition inferenceWritePart = new Partition(3);
		
		Set<StandardPredicate> closedPredicates = new HashSet<StandardPredicate>(Arrays.asList(X));
		
		// Insert Z
		boolean ContainsZ = checkContainsLatentVariable();
		if (ContainsZ) {
			Inserter insert = data.getInserter(Z[0], targetPart);
			Inserter insert2 = data.getInserter(Z[0], inferenceWritePart);
			Set<GroundAtom> groundings = Queries.getAllAtoms(wlTruthDB, Y[0]);
			for (GroundAtom ga : groundings) {
				GroundTerm[] arguments = ga.getArguments();
				insert.insert((Object[]) arguments);
				insert2.insert((Object[]) arguments);
			}
		}
		
		Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
		Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
		
		// Insert Y
		ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(Y[0]));
		for (int i=0; i<allGroundings.size(); i++) {
			GroundTerm [] grounding = allGroundings.get(i);
			GroundAtom atom = inferenceDB.getAtom(Y[0], grounding);
			GroundAtom atom1 = wlTrainDB.getAtom(Y[0], grounding);
			if (atom instanceof RandomVariableAtom) {
				wlTrainDB.commit((RandomVariableAtom) atom1);
				inferenceDB.commit((RandomVariableAtom) atom);
			}
		}
		
		// Calculate Semantic Distances
		double[] dists = calculateSemanticDist();
		
		// Do weight Learning
		WeightLearningApplication weightLearner = null;
		if (ContainsZ) { // Z.length>0
//			weightLearner = new HardEM(model, wlTrainDB, wlTruthDB, config);
			weightLearner = new SemanticHardEM(model, wlTrainDB, wlTruthDB, dists, config);
		} else {
			weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
//			weightLearner = new SemanticMaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, dists, config);
		}
	    ArrayList<Integer> numGroundedList = weightLearner.learn();
	    weightLearner.close();
	    wlTrainDB.close();
	    data.deletePartition(targetPart);
	    
	    double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();

		// Do Inference
		MPEInference mpe = new MPEInference(model, inferenceDB, config);
		FullInferenceResult result = mpe.mpeInference();
		mpe.close();
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
//		
//		if (listSize != 0)
//			loss_len = -sumLen*1.0/listSize;
//		loss_num = -listSize;
//
//		double reward = LAMBDA_ruleLen* loss_len; 
//		reward += LAMBDA_ruleNum* loss_num;
//		return reward;
//	}
	
	public double[] calculateSemanticDist() { // Distance to Right Reasons
		int kernelSize = 0;
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class))
			kernelSize++;
		
		double[] dists = new double[kernelSize];
		final double satisfy_dist = 0.0;
		final double notSatisfy_dist = 1.0;
		
		for (int i=0; i<kernelSize; i++) {			
			boolean EXIST_BULLYINGSIGNAL = false;
			boolean EXIST_NONBULLYINGSIGNAL = false;
			
			boolean TARGET_POSITIVE_HEAD; // bullyings
			int target_idx;
			if (ruleListEmbedding[i][posPredNum-2]==1) { // Positive Y in Body
				target_idx = posPredNum-2;
				TARGET_POSITIVE_HEAD = false;
			} else if (ruleListEmbedding[i][posPredNum+ negPredNum-2]==1) { // Negative Y in Body
				target_idx = posPredNum+ negPredNum-2;
				TARGET_POSITIVE_HEAD = true;
			} else if (ruleListEmbedding[i][posPredNum-1]==1) { // Positive Z in Body
				target_idx = posPredNum-1;
				TARGET_POSITIVE_HEAD = true;
			} else { // Negative Z in Body
				target_idx = posPredNum+negPredNum-1;
				assert ruleListEmbedding[i][target_idx]==1 : "No Target in Rule";
				TARGET_POSITIVE_HEAD = false;
			}
//			if (ruleListEmbedding[i][posPredNum-1]==1) {
//				target_idx = posPredNum-1;
//				TARGET_POSITIVE_HEAD = false;
//			} else {
//				target_idx = posPredNum+negPredNum-1;
//				assert ruleListEmbedding[i][target_idx]==1 : "No Target in Rule";
//				TARGET_POSITIVE_HEAD = true;
//			}
			
			for (int j=0; j<ReturnAction; j++) {
				if (ruleListEmbedding[i][j]==1 && j!=target_idx) {
					if (j < posPredNum) { // Positive 
						String p_name = allPosPreds[j].getName();
						if (Bullying_Signal.contains(p_name)) 
							EXIST_BULLYINGSIGNAL = true;
						else if (Nonbullying_Signal.contains(p_name))
							EXIST_NONBULLYINGSIGNAL = true;
					} else { // Negative
						String p_name = "~"+ allNegPreds[j-posPredNum].getName();
						if (Bullying_Signal.contains(p_name))
							EXIST_BULLYINGSIGNAL = true;
						else if (Nonbullying_Signal.contains(p_name))
							EXIST_NONBULLYINGSIGNAL = true;
					}
				}
			}
			
			// Calculate Distance to Right Reasons
			if (EXIST_BULLYINGSIGNAL && EXIST_NONBULLYINGSIGNAL)
				dists[i] = satisfy_dist;
			else if (!EXIST_BULLYINGSIGNAL && !EXIST_NONBULLYINGSIGNAL)
				dists[i] = satisfy_dist;
			else if (EXIST_BULLYINGSIGNAL && !EXIST_NONBULLYINGSIGNAL) {
				if (TARGET_POSITIVE_HEAD)
					dists[i] = satisfy_dist;
				else
					dists[i] = notSatisfy_dist;
			}
			else {
				if (TARGET_POSITIVE_HEAD)
					dists[i] = notSatisfy_dist;
				else
					dists[i] = satisfy_dist;
			}
		}
		return dists;
	}
	
	public boolean buildNewRule(int[] ruleEmbedding) {
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		final double initWeight = 5.0;
		
		// Add Dummy Predicates to body
		// First Add User Dummy Predicate
		List<GenericVariable> argsList = generalPredArgsMap.get(UserDummyPred);
		Object[] args = new Object[argsList.size()];
		args = argsList.toArray(args);
		body = (FormulaContainer) model.createFormulaContainer(UserDummyPred.getName(), args);
		
		// Randomly Choose Head Predicate
		List<Integer> potentialHeadPreds = new ArrayList<Integer>();
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i] == 1) {
				potentialHeadPreds.add(i);
			}
		}
		int headIdx = potentialHeadPreds.get(rdm.nextInt(potentialHeadPreds.size()));
		if (headIdx < posPredNum) { // Negative Head
			StandardPredicate p = allPosPreds[headIdx];
			argsList = generalPredArgsMap.get(p);
			args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			head = (FormulaContainer) head.bitwiseNegate();
		} else { // Positive Head
			StandardPredicate p = allNegPreds[headIdx-posPredNum];
			argsList = generalPredArgsMap.get(p);
			args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
		}

		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1 && i!=headIdx) {
				if (i<posPredNum) { // Positive Body Predicate
					StandardPredicate p = allPosPreds[i];
					argsList= generalPredArgsMap.get(p);
					args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null)
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Negative Body Predicate
					StandardPredicate p = allNegPreds[i-posPredNum];
					argsList= generalPredArgsMap.get(p);
					args = new Object[argsList.size()];
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

