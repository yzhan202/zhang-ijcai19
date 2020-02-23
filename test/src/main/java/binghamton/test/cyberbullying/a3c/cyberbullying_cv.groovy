package binghamton.test.cyberbullying.a3c

import binghamton.rl_latent.A3C.generator_latent.a3cMDP_latent
import binghamton.rl_latent.A3C.latentPSLModelCreation
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate


def fold = 0; //2; // 0, 1
def outputPath = 'result/crossValidation_bullying/'+fold+'/'


/*
 * Training
 */
def mdp = new a3cMDP_latent();

// Load
//mdp.loadA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser')


mdp.train();

println("Finish training")

/*
 * Save Result
 */
File outputDir = new File(outputPath);
if (!outputDir.exists()) {
	outputDir.mkdirs();
}
mdp.saveA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser');




