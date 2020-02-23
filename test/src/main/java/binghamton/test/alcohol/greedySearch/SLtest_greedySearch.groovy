package binghamton.test.alcohol.greedySearch;

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
import binghamton.greedySearch.greedySearch
import binghamton.greedySearch.greedySearch_network
import binghamton.rl.DQN.generator1.MDP_NN
import binghamton.rl.DQN.generator1.dqnMDP_NN
import binghamton.rl.DQN.generator2.dqnMDP_NN2
import binghamton.rl.DQN.hindsight.dqnMDP_hindsight
import binghamton.util.DataOutputter;


def fold = 3;
String dataDir = 'data/crossValidation/alcohol_data/'+fold+'/training'; //'data/alcohol_test'; //'data/smallAlcohol';

//config manager
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("SL_alcohol")
Logger log = LoggerFactory.getLogger(this.class)

//database
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "structureLearning_alcohol")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel model = new PSLModel(this, data)

private List<String> getAlcoholWords(){
	def fullFilePath = "data/topAlcoholWords.txt";
	BufferedReader br = new BufferedReader(new FileReader(fullFilePath));
	List<String> topAlcoholWords = new ArrayList<String>();

	try {
		String line = br.readLine();
		while (line != null) {
			topAlcoholWords.add(line)
			line = br.readLine();
		}
	} finally {
		br.close();
	}
	return topAlcoholWords;
}


model.add predicate: "recovers", types: [ArgumentType.UniqueID]
model.add predicate: "attendsAA", types: [ArgumentType.UniqueID]
model.add predicate: "localRecovers", types: [ArgumentType.UniqueID]
model.add predicate: "usesAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "usesSoberWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "containsAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "containsSoberWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "alcoholTopic", types: [ArgumentType.UniqueID]
model.add predicate: "soberTopic", types: [ArgumentType.UniqueID]
////model.add predicate: "affect", types: [ArgumentType.UniqueID]
////model.add predicate: "social", types: [ArgumentType.UniqueID]

// Friend Feature
model.add predicate: "friends", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendContainsAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendContainsSoberWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendUsesAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendUsesSoberWord", types: [ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "replies", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "retweets", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendReplies", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendRetweets", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]


//def closedPredicates = [attendsAA, containsAlcoholWord, containsSoberWord, localRecovers, usesAlcoholWord, usesSoberWord,
//	alcoholTopic, soberTopic] as Set;
//def closedPredicates = [friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord,
//	replies, retweets, friendReplies, friendRetweets] as Set;

def closedPredicates = [attendsAA, containsAlcoholWord, containsSoberWord, localRecovers, usesAlcoholWord, usesSoberWord,
	alcoholTopic, soberTopic, friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord,
	replies, retweets, friendReplies, friendRetweets] as Set;

def inferredPredicates = [recovers] as Set;

//StandardPredicate[] negFeatPreds = [localRecovers, usesAlcoholWord, usesSoberWord, alcoholTopic, soberTopic, affect, social]
StandardPredicate[] negFeatPreds = [localRecovers, alcoholTopic, soberTopic];

StandardPredicate[] friendPreds = [friends, friendContainsAlcoholWord, friendContainsSoberWord,
	friendUsesAlcoholWord, friendUsesSoberWord, replies, retweets, friendReplies, friendRetweets];
//StandardPredicate[] networkPreds = [replies, retweets, friends, friendReplies, friendRetweets];
//StandardPredicate[] friendPreds = [];
//StandardPredicate[] networkPreds = [];

def predicateFileMap = [((Predicate)recovers):"recovers.txt",
	((Predicate)attendsAA): "attendsAA.txt",
	((Predicate)localRecovers):"localRecovers.txt",
	((Predicate)usesAlcoholWord):"usesAlcoholWord.txt",
	((Predicate)usesSoberWord):"usesSoberWord.txt",
	((Predicate)containsAlcoholWord):"containsAlcoholWord.txt",
	((Predicate)containsSoberWord):"containsSoberWord.txt",
	((Predicate)alcoholTopic):"alcoholTopic.txt",
	((Predicate)soberTopic):"soberTopic.txt",
////	((Predicate)affect):"affect.txt",
////	((Predicate)social):"social.txt"]
	
	((Predicate)friends):"friends.txt",
	((Predicate)friendContainsAlcoholWord):"friendContainsAlcoholWord.txt",
	((Predicate)friendContainsSoberWord):"friendContainsSoberWord.txt",
	((Predicate)friendUsesAlcoholWord):"friendUsesAlcoholWord.txt",
	((Predicate)friendUsesSoberWord):"friendUsesSoberWord.txt",
	((Predicate)replies):"replies.txt",
	((Predicate)retweets):"retweets.txt",
	((Predicate)friendReplies):"friendReplies.txt",
	((Predicate)friendRetweets):"friendRetweets.txt"]
	

def predicateSoftTruthMap = [((Predicate)recovers):true,
	((Predicate)attendsAA):false,
	((Predicate)localRecovers):true,
	((Predicate)usesAlcoholWord):false,
	((Predicate)usesSoberWord):false,
	((Predicate)containsAlcoholWord):false,
	((Predicate)containsSoberWord):false,
	((Predicate)alcoholTopic):false,
	((Predicate)soberTopic):false,
////	((Predicate)affect):true,
////	((Predicate)social):true]
	
	((Predicate)friends):false,
	((Predicate)friendContainsAlcoholWord):false,
	((Predicate)friendContainsSoberWord):false,
	((Predicate)friendUsesAlcoholWord):false,
	((Predicate)friendUsesSoberWord):false,
	((Predicate)replies):false,
	((Predicate)retweets):false,
	((Predicate)friendReplies):false,
	((Predicate)friendRetweets):false]
	

GenericVariable U = new GenericVariable('U', model);
GenericVariable AW = new GenericVariable('AW', model);
GenericVariable SW = new GenericVariable('SW', model);
GenericVariable I = new GenericVariable('I', model);
// Arguments for Friends
GenericVariable U1 = new GenericVariable('U1', model);
GenericVariable I1 = new GenericVariable('I1', model);
GenericVariable AW1 = new GenericVariable('AW1', model);
GenericVariable SW1 = new GenericVariable('SW1', model);

def generalPredArgsMap = [((Predicate)recovers): [U],
	((Predicate)attendsAA): [U],
	((Predicate)localRecovers): [U],
	((Predicate)usesAlcoholWord): [U,AW],
	((Predicate)usesSoberWord): [U,SW],
	((Predicate)containsAlcoholWord): [U,I,AW],
	((Predicate)containsSoberWord): [U,I,SW],
////	((Predicate)affect): [U],
////	((Predicate)social): [U],
	((Predicate)alcoholTopic): [U],
	((Predicate)soberTopic): [U],
	
	((Predicate)friends): [U,U1],
	((Predicate)friendContainsAlcoholWord): [U1,I1,AW1],
	((Predicate)friendContainsSoberWord): [U1,I1,SW1],
	((Predicate)friendUsesAlcoholWord): [U1,AW1],
	((Predicate)friendUsesSoberWord): [U1,SW1],
	((Predicate)replies): [U,U1,I],
	((Predicate)retweets): [U,U1,I],
	((Predicate)friendReplies): [U1,U,I1],
	((Predicate)friendRetweets): [U1,U,I1]]


def topAlcoholWords = getAlcoholWords();
def topSoberWords = ['#recovery', 'sober', 'sobriety', 'recovery', '#sobriety'];
//def specificPredArgsMap = [
//	((Predicate)usesAlcoholWord): [[U],topAlcoholWords],
//	((Predicate)usesSoberWord): [[U],topSoberWords],
//	((Predicate)containsAlcoholWord): [[U],[I],topAlcoholWords],
//	((Predicate)containsSoberWord): [[U],[I],topSoberWords],
	
//	((Predicate)friendUsesAlcoholWord): [[U1],topAlcoholWords],
//	((Predicate)friendUsesSoberWord): [[U1],topSoberWords],
//	((Predicate)friendContainsAlcoholWord): [[U1],[I1],topAlcoholWords],
//	((Predicate)friendContainsSoberWord): [[U1],[I1],topSoberWords]
//]


Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
//Partition targetPart = new Partition(2);
//Partition inferenceWritePart = new Partition(3);

def inserter;
for (Predicate p: closedPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir + '/' + fileName;
	println p.toString()
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: [recovers]) {
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

//Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
//populateDatabase(data, wlTrainDB, truthPart, inferredPredicates);

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

//Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
//populateDatabase(data, inferenceDB, truthPart, inferredPredicates);

//def allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(recovers));
//for (int i=0; i<allGroundings.size(); i++) {
//	GroundTerm [] grounding = allGroundings.get(i)
//	GroundAtom atom = inferenceDB.getAtom(recovers, grounding);
//	if (atom instanceof RandomVariableAtom) {
//		inferenceDB.commit((RandomVariableAtom) atom);
//	}
//}


StandardPredicate[] X = closedPredicates.toArray();
StandardPredicate[] Y = inferredPredicates.toArray();
StandardPredicate[] Z = [];

/*
 * Greedy Search Starts
 */
//def mySearch = new greedySearch(X, Y, Z, model, data, wlTruthDB, trainPart, config, generalPredArgsMap, null, negFeatPreds);

def mySearch = new greedySearch_network(X, Y, Z, model, data, wlTruthDB, trainPart, config, generalPredArgsMap, null, negFeatPreds, friendPreds);

mySearch.search();

