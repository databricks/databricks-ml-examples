package com.example.app;

import com.databricks.ml.local.LocalModel;
import com.databricks.ml.local.ModelFactory;

/** Example of loading and using a Pipeline taking Vector input. */
public class AppVectorInput {
  public static void main(String[] args) {
    // The path to our example logistic regression model: update this path to load your own
    // models. In practice, models could be in a resources directory or on an accessible
    // filesystem.
    String path = "my_models/lr_pipeline";
    LocalModel model = ModelFactory.loadModel(path);
    System.out.println(model);

    // Select output columns which the model should return.
    // Here, we output the new "prediction" and "probability" columns
    // and keep the existing "label" column.
    // The "probability" column contains a Vector of predicted probabilities for each class.
    String[] outputs = {"label", "prediction", "probability"};
    model.setOutputCols(outputs);

    // The model input is a standard JSON string.
    // The input schema here is: [origLabel: Double, features: Vector].
    String input =
        "{\"origLabel\":-1.0,"
            + "\"features\":{\"type\":0,\"size\":13,"
            + "\"indices\":[0,2,3,4,6,7,8,9,10,11,12],"
            + "\"values\":[74.0,2.0,120.0,269.0,2.0,121.0,1.0,0.2,1.0,1.0,3.0]}"
            + "}";
    System.out.println("INPUT: " + input);

    // The model output is also a standard JSON string, with the expected output fields.
    String output = model.transform(input);
    System.out.println("OUTPUT: " + output);
  }
}
