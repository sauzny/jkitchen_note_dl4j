package com.sauzny.mnist;

import java.io.File;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public final class SerializableModelUtils {

    private SerializableModelUtils(){}
    
    public static void out(MultiLayerNetwork model){
        out(model, "D://LungCNNModel.zip");
    }
    
    public static MultiLayerNetwork in(){
        return in("D://LungCNNModel.zip");
    }
    
    public static void out(MultiLayerNetwork model, String modelPath) {
        try {
            //save model  
            File locationToSave = new File(modelPath);  
            boolean saveUpdater = true;  
            ModelSerializer.writeModel(model, locationToSave, saveUpdater); 
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static MultiLayerNetwork in(String modelPath){
        MultiLayerNetwork model = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(modelPath);  
        } catch (Exception e) {
            e.printStackTrace();
        }
        return model;
    }
}
