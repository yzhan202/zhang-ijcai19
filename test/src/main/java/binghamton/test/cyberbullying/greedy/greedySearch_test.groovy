package binghamton.test.bullying;

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
import binghamton.greedySearch.greedySearch_bullying
import binghamton.util.DataOutputter;


def fold = 4;
String dataDir = 'data/cyberbullying/bullying_cv5/'+fold+'/training';

//config manager
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("SL_bullying")
Logger log = LoggerFactory.getLogger(this.class)

// database
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "structureLearning_alcohol")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel model = new PSLModel(this, data)

model.add predicate: "bullyings", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "anonymity", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "containsBullyingWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "posSentiment", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "anger", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "nego", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "sexTopic", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "hatredTopic", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

model.add predicate: "questionAnswer", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

def closedPredicates = [anonymity, containsBullyingWord, posSentiment, 
	anger, nego, sexTopic, hatredTopic, questionAnswer] as Set
def inferredPredicates = [bullyings] as Set

StandardPredicate[] negFeatPreds = [anonymity, posSentiment];

def predicateFileMap = [((Predicate)bullyings):"bullyings.txt",
	((Predicate)anonymity):"anonymity.txt",
	((Predicate)containsBullyingWord):"containsBullyingWord.txt",
	((Predicate)posSentiment):"posSentiment.txt",
	((Predicate)anger):"anger.txt",
	((Predicate)nego):"nego.txt",
	((Predicate)sexTopic):"sexTopic.txt",
	((Predicate)hatredTopic):"hatredTopic.txt",
	
	((Predicate)questionAnswer):"questionAnswer.txt"]


def predicateSoftTruthMap = [((Predicate)bullyings):true,
	((Predicate)anonymity):true,
	((Predicate)containsBullyingWord):false,
	((Predicate)posSentiment):true,
	((Predicate)anger):true,
	((Predicate)nego):true,
	((Predicate)sexTopic):false,
	((Predicate)hatredTopic):false,
	((Predicate)questionAnswer):false]


GenericVariable U = new GenericVariable('U', model);
GenericVariable I = new GenericVariable('I', model);
GenericVariable BW = new GenericVariable('BW', model);

def generalPredArgsMap = [((Predicate)bullyings): [U,I],
	((Predicate)anonymity): [U,I],
	((Predicate)containsBullyingWord): [U,I,BW],
	((Predicate)posSentiment): [U,I],
	((Predicate)anger): [U,I],
	((Predicate)nego): [U,I],
	((Predicate)sexTopic): [U,I],
	((Predicate)hatredTopic): [U,I],
	((Predicate)questionAnswer): [U,I]]

Partition trainPart = new Partition(0);
Partition truthPart = new Partition(1);

def inserter;
for (Predicate p : closedPredicates) {
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

for (Predicate p : [bullyings]) {
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

Database wlTruthDB = data.getDatabase(truthPart, inferredPredicates);

StandardPredicate[] X = closedPredicates.toArray();
StandardPredicate[] Y = inferredPredicates.toArray();
StandardPredicate[] Z = [];

def mySearch = new greedySearch_bullying(X, Y, Z, model, data, wlTruthDB, trainPart, config, generalPredArgsMap, null, negFeatPreds);
mySearch.search();





