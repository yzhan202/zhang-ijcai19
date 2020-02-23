package binghamton.test.cyberbullying.a3c

import java.text.DecimalFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*;
import edu.umd.cs.psl.core.*;
import edu.umd.cs.psl.core.inference.*;
import edu.umd.cs.psl.database.*;
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.atom.*;
import edu.umd.cs.psl.model.formula.*;
import edu.umd.cs.psl.model.function.*;
import edu.umd.cs.psl.model.kernel.*;
import edu.umd.cs.psl.model.kernel.rule.AbstractRuleKernel
import edu.umd.cs.psl.model.kernel.rule.CompatibilityRuleKernel
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*;
import edu.umd.cs.psl.util.database.*;
import com.google.common.collect.Iterables;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.evaluation.resultui.printer.*;
import java.io.*;
import java.util.*;
import groovy.time.*;

import binghamton.util.FoldUtils
import binghamton.util.GroundingWrapper
import binghamton.rl.DQN.generator1.MDP_NN
import binghamton.rl.DQN.generator1.dqnMDP_NN
import binghamton.rl.DQN.generator2.dqnMDP_NN2
import binghamton.rl.DQN.hindsight.dqnMDP_hindsight
import binghamton.util.DataOutputter;

import org.apache.commons.lang3.ArrayUtils



def fold = 1;
String dataDir = 'data/crossValidation/bullying_cv5/'+ fold+ '/test';

//config manager
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("SL_bullying")
Logger log = LoggerFactory.getLogger(this.class)

//database
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "structureLearning_bullying")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel model = new PSLModel(this, data)

model.add predicate: "bullyings", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "questionAnswer", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

model.add predicate: "containsBullyingWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "posSentiment", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "anonymity", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "nego", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "anger", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "sexTopic", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "hatredTopic", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]


def closedPredicates = [questionAnswer, containsBullyingWord, posSentiment, anonymity,
			nego, anger, sexTopic, hatredTopic] as Set;
def inferredPredicates = [bullyings] as Set;
StandardPredicate[] negFeatPreds = [posSentiment, anonymity]

def predicateFileMap = [((Predicate)bullyings):"bullyings.txt",
			((Predicate)questionAnswer):"questionAnswer.txt",
			((Predicate)containsBullyingWord):"containsBullyingWord.txt",
			((Predicate)posSentiment):"posSentiment.txt",
			((Predicate)anonymity):"anonymity.txt",
			((Predicate)nego):"nego.txt",
			((Predicate)anger):"anger.txt",
			((Predicate)sexTopic):"sexTopic.txt",
			((Predicate)hatredTopic):"hatredTopic.txt"]
			
		
def predicateSoftTruthMap = [((Predicate)bullyings):true,
	((Predicate)questionAnswer):false,
	((Predicate)containsBullyingWord):false,
	((Predicate)posSentiment):true,
	((Predicate)anonymity):true,
	((Predicate)nego):true,
	((Predicate)anger):true,
	((Predicate)sexTopic):false,
	((Predicate)hatredTopic):false]

GenericVariable U = new GenericVariable('U', model);
GenericVariable I = new GenericVariable('I', model);
GenericVariable BW = new GenericVariable('BW', model);

def generalPredArgsMap = [((Predicate)bullyings): [U,I],
	((Predicate)questionAnswer): [U,I],
	((Predicate)containsBullyingWord): [U,I,BW],
	((Predicate)posSentiment): [U,I],
	((Predicate)anonymity): [U,I],
	((Predicate)nego): [U,I],
	((Predicate)anger): [U,I],
	((Predicate)sexTopic): [U,I],
	((Predicate)hatredTopic): [U,I]]

/*
 * Parse PSL Rules from Text File
 */
boolean isPredicateString(String str) {
	if (str.contains(")") || str.contains(",") || str=="")
		return false;
	else
		return true;
}
def modelSize = 0;
String modelFile = 'result/MLNs/bullying/'+fold+'/L1Model_04292019.txt'
BufferedReader reader = new BufferedReader(new FileReader(modelFile));
String read = null;
while ((read = reader.readLine()) != null) {
	FormulaContainer body = null;
	FormulaContainer head = null;
	FormulaContainer rule = null;
	
	boolean LOGIC_NOT = false;
	String[] splited = read.split(">>");
	String bodyPart = splited[0];
	String headPart = splited[1];
	String[] head_splited = (((headPart.split("\\{"))[0]).replace(" ", "")).split("\\(");
	for (int i=0; i<head_splited.length; i++) {
		if (head_splited[i].contains("~")) {
			LOGIC_NOT = true;
		} else if (isPredicateString(head_splited[i])) {
			StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(head_splited[i]); //
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			
			if (LOGIC_NOT) {
				head = (FormulaContainer) head.bitwiseNegate();
				LOGIC_NOT = false;
			}
		}
	}
	
	String weightPart = bodyPart.split("\\}")[0];
	bodyPart = bodyPart.split("\\}")[1].replace(" ", "");
	double weightValue = Double.parseDouble(weightPart.replace("{", ""));
	
	if (weightValue < 2.0)
		continue
	
	String[] body_splited = bodyPart.split("&");
	for (int i=0; i<body_splited.length; i++) {
		String[] tmp = body_splited[i].split("\\(");
		for (int j=0; j<tmp.length; j++) {
			if (tmp[j].contains("~")) {
				LOGIC_NOT = true;
			} else if (isPredicateString(tmp[j])) {
				StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(tmp[j]);
				List<GenericVariable> argsList = generalPredArgsMap.get(p);
				Object[] args = new Object[argsList.size()];
				args = argsList.toArray(args);
				
				if (body == null) {
					body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						body = (FormulaContainer) body.bitwiseNegate();
						LOGIC_NOT = false;
					}
				} else {
					FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						LOGIC_NOT = false;
					}
					body = (FormulaContainer) body.and(f_tmp);
				}
			}
		}
	}
	
	rule = (FormulaContainer) body.rightShift(head);
	Map<String, Object> argsMap = new HashMap<String, Object>();
	argsMap.put("rule", rule);
	argsMap.put("sqaured", true);
	argsMap.put("weight", weightValue);
	
	model.add(argsMap);
	modelSize++;
	
//	if (modelSize > 25)
//		break;
}
reader.close();

System.out.println(model.toString());

Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
Partition inferenceWritePart = new Partition(2);

def inserter;
for (Predicate p: closedPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir + '/' + fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: inferredPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir + '/' + fileName;
	if(predicateSoftTruthMap[p]){
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	}
	else{
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
	Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
	DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);

	for (Predicate p : inferredPredicates){
		dbPop.populateFromDB(populationDatabase, p);
	}
	populationDatabase.close();
}

Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
populateDatabase(data, inferenceDB, truthPart, inferredPredicates);

/*
 * Inference
 */
println("Doing Inference")
MPEInference mpe = new MPEInference(model, inferenceDB, config)
FullInferenceResult result = mpe.mpeInference()
inferenceDB.close();

Database resultsDB = data.getDatabase(inferenceWritePart, inferredPredicates)

def comparator = new SimpleRankingComparator(resultsDB)
def groundTruthDB = data.getDatabase(truthPart, inferredPredicates)
def list = Queries.getAllAtoms(groundTruthDB, bullyings);

comparator.setBaseline(groundTruthDB)

def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
double [] score = new double[metrics.size()]
for (int i = 0; i < metrics.size(); i++) {
	comparator.setRankingScore(metrics.get(i))
	score[i] = comparator.compare(bullyings)
}

System.out.println("Area under positive-class PR curve: " + score[0])
System.out.println("Area under negative-class PR curve: " + score[1])
System.out.println("Area under ROC curve: " + score[2])




