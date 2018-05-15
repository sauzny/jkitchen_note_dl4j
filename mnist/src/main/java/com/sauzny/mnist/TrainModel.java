package com.sauzny.mnist;

import java.io.IOException;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class TrainModel {

    public static void main(String[] args) throws IOException {
        
        long start = System.currentTimeMillis();
        
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        
        /**
         * batchSize 和 numEpochs必须根据经验选择，而经验则需要通过实验来积累。
         * batchSize，每批次处理的数据越多，训练速度越快；
         * epoch的数量越多，遍历数据集的次数越多，准确率就越高。
         * 但是，epoch的数量达到一定的大小之后，增益会开始减少，所以要在准确率与训练速度之间进行权衡。
         * 总的来说，需要进行实验才能发现最优的数值。
         * 本示例中设定了合理的默认值。
         */
        
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 15;
        
        //Get the DataSetIterators:
        DataSetIterator mnistTrain = TrainModelUtils.mnistTrain();
        DataSetIterator mnistTest = TrainModelUtils.mnistTest();

        log.info("Build model....");
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                    .nIn(height * width)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(100)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            model.fit(mnistTrain);
        }
        
        // 保存模型
        SerializableModelUtils.out(model);
        
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
        
        long end = System.currentTimeMillis();
        
        System.out.println(end - start);
    }
}

/**
指示符                         描述
Accuracy    准确率：模型准确识别出的MNIST图像数量占总数的百分比。
Precision   精确率：真正例的数量除以真正例与假正例数之和。
Recall      召回率：真正例的数量除以真正例与假负例数之和。
F1 Score    F1值  ：精确率和召回率的加权平均值。
*/

/*
 * 精确率、召回率和F1值衡量的是模型的相关性。
 * 举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，因为病人会不再寻求进一步治疗。
 * 所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），尽管总体上的准确率可能会相对较低一些。
 */
