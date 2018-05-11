package com.sauzny.mnist;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFileChooser;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.spark_project.guava.collect.Lists;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class MyMnist01 {

    public static String fileChose() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
          File file = fc.getSelectedFile();
          String filename = file.getAbsolutePath();
          return filename;
        } else {
          return null;
        }
      }
    
    public static void main(String[] args) throws IOException {
        int height = 28;
        int width = 28;
        int channels = 1;

        // recordReader.getLabels()
        // In this version Labels are always in order
        // So this is no longer needed
        //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
        List<Integer> labelList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        // pop up file chooser
        String filechose = fileChose().toString();

        //LOAD NEURAL NETWORK

        // Where to save model
        //File locationToSave = new File(DATA_PATH + "trained_mnist_model.zip");
        // Check for presence of saved model
        /*
        if (locationToSave.exists()) {
          log.info("Saved Model Found!");
        } else {
          log.error("File not found!");
          log.error("This example depends on running MnistImagePipelineExampleSave, run that example first");
          System.exit(0);
        }
         */
        MultiLayerNetwork model = SerializableModelUtils.in();

        log.info("TEST YOUR IMAGE AGAINST SAVED NETWORK");
        // FileChose is a string we will need a file
        File file = new File(filechose);

        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray
        INDArray image = loader.asMatrix(file);
        
        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        
        // Pass through to neural Net
        INDArray output = model.output(image.reshape(1, 784));

        log.info("The file chosen was " + filechose);
        log.info("The neural nets prediction (list of probabilities per label)");
        //log.info("## List of Labels in Order## ");
        // In new versions labels are always in order
        log.info(output.toString());
        log.info(labelList.toString());
        
        INDArray result = output.reshape(10);
        
        Double max = result.maxNumber().doubleValue();
        
        List<Integer> maxIndex = Lists.newArrayList();
        
        for(int i=0; i<result.toDoubleVector().length; i++){
            if(max == result.toDoubleVector()[i]){
                maxIndex.add(i);
            }
        }
        log.info("图片中的数字为：{}", maxIndex);
    }
}
