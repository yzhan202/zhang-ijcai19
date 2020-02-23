package binghamton.rl.A3C;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import binghamton.rl.NeuralNet;

import java.io.IOException;
import java.io.OutputStream;


public interface IActorCritic<NN extends IActorCritic> extends NeuralNet<NN> {
    boolean isRecurrent();

    void reset();

    void fit(INDArray input, INDArray[] labels);

    //FIRST SHOULD BE VALUE AND SECOND IS SOFTMAX POLICY. DONT MESS THIS UP OR ELSE ASYNC THREAD IS BROKEN (maxQ) !
    INDArray[] outputAll(INDArray batch);

    NN clone();

    void copy(NN from);

    Gradient[] gradient(INDArray input, INDArray[] labels);

    void applyGradient(Gradient[] gradient, int batchSize);

    void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException;

    void save(String pathValue, String pathPolicy) throws IOException;

    double getLatestScore();

}
