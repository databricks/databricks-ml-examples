package com.example.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.databricks.ml.local.LocalModel;
import com.databricks.ml.local.ModelFactory;


/**
 * Example of loading and using a Pipeline taking Vector input in a multi-threaded environment. We
 * recommend loading and configuring the model in a single thread and running the inference via
 * calling the `model.transform` method across a dataset in worker threads.
 *
 * <p>Please don't mutate loaded model in worker threads.
 */
public class AppMultiThreading {
  /**
   * A task data container sortable by their integer ID. The sortable IDs are only used for result
   * validation purposes. They are not necessary for running inference tasks.
   */
  private static class TaskRecord implements Comparable<TaskRecord> {
    final int id;
    final String data;

    /**
     * Create a TaskRecord.
     *
     * @param id an integer index for the task
     * @param data a JSON encoded example
     */
    TaskRecord(int id, String data) {
      this.id = id;
      this.data = data;
    }

    @Override
    public String toString() {
      return "ID: " + this.id + ", DATA: " + this.data;
    }

    public int compareTo(TaskRecord other) {
      return this.id - other.id;
    }
  }

  /**
   * A class to wrap around a single model inference task, so that it can be executed by a worker
   * thread. This is a minimal example of how dbml-local can be used in a multi-threaded
   * environment.
   */
  private static class InferenceTask implements Callable<List<TaskRecord>> {
    private final LocalModel model;
    private final List<TaskRecord> inputs;

    /**
     * Create a new instance of InferenceTask
     *
     * @param model an initialized LocalModel instance.
     * @param inputs a list of TaskRecords that contain JSON encoded string of the input features
     *     and the integer IDs of these examples.
     */
    InferenceTask(LocalModel model, List<TaskRecord> inputs) {
      this.model = model;
      this.inputs = inputs;
    }

    public List<TaskRecord> call() {
      List<TaskRecord> inferenceResults = new ArrayList<TaskRecord>();
      for (TaskRecord input : this.inputs) {
        // The model output is also a standard JSON string, with the expected output fields.
        TaskRecord result = new TaskRecord(input.id, model.transform(input.data));
        inferenceResults.add(result);
      }
      return inferenceResults;
    }
  }

  /** Create JSON encoded random features. Each feature is a sparse vector of Doubles. */
  private static class FeatureGenerator {
    private final int featureSize;

    /**
     * Create new features as a vectors of a certain length.
     *
     * @param featureSize length of the feature vector.
     */
    FeatureGenerator(int featureSize) {
      this.featureSize = featureSize;
    }

    List<TaskRecord> newFeatures(int numTasks) {
      List<TaskRecord> inputs = new ArrayList<TaskRecord>();
      for (int idx = 0; idx < numTasks; ++idx) {
        inputs.add(new TaskRecord(idx, this.newJsonFeature(idx + 0.5)));
      }
      return inputs;
    }

    private String newJsonFeature(double shift) {
      // The model input is a standard JSON string.
      // The input schema here is: [origLabel: Double, features: Vector].
      List<Double> features = new ArrayList<Double>();
      for (int i = 0; i < this.featureSize; ++i) {
        features.add(i + shift);
      }
      return "{\"origLabel\":-1.0,"
          + "\"features\":{\"type\":1,"
          + "\"values\":"
          + features.toString()
          + "}"
          + "}";
    }
  }

  // Create a thread pool with fixed size.
  private static final long singleTaskTimeout = 10L; // in seconds
  private static final long executorShutdownTimeout = 10L; // in seconds

  /**
   * Perform parallel inference task on inputs. Each task runs inference on a batch of examples.
   *
   * @param model a LocalModel instance.
   * @param inputs a list of JSON encoded feature samples.
   * @param threadPoolSize number of threads to run inference tasks in parallel.
   * @param batchSize the number of samples each inference task carries.
   */
  private static List<TaskRecord> runTestWithParallelism(
      LocalModel model, List<TaskRecord> inputs, int threadPoolSize, int batchSize)
      throws InterruptedException {
    // Create individual inferences tasks.
    final int numTasks = inputs.size();
    List<InferenceTask> tasks = new ArrayList<InferenceTask>();
    for (int batchBegin = 0; batchBegin < numTasks; batchBegin += batchSize) {
      int batchEnd = Math.min(batchBegin + batchSize, numTasks);
      List<TaskRecord> batchedInputs = inputs.subList(batchBegin, batchEnd);
      tasks.add(new InferenceTask(model, batchedInputs));
    }

    ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);

    try {
      // Dispatch the tasks to the threads.
      List<TaskRecord> results = new ArrayList<TaskRecord>();

      for (Future<List<TaskRecord>> future : executor.invokeAll(tasks)) {
        try {
          List<TaskRecord> batchResults = future.get(singleTaskTimeout, TimeUnit.SECONDS);
          results.addAll(batchResults);
        } catch (TimeoutException ex) {
          throw new RuntimeException(
              "one inference task timed out after " + singleTaskTimeout + " seconds");
        } catch (CancellationException ce) {
          throw new RuntimeException("one inference task canceled. " + ce.getMessage());
        } catch (ExecutionException ee) {
          throw new RuntimeException(
              "one inference task failed to execute. " + ee.getCause().getMessage());
        }
      }

      // Do something with the results
      if (!results.isEmpty()) {
        System.out.println("------------ TASK INFO ------------");
        System.out.println(">> parallelism " + threadPoolSize);
        System.out.println(">> batch size  " + batchSize);
        System.out.println("INPUT[0]:" + inputs.get(0));
        System.out.println("OUTPUT[*]: Received " + results.size() + " results");
        System.out.println("OUTPUT[0]: " + results.get(0));
      }
      return results;

    } finally {
      // Shutdown the thread pool after all tasks are done.
      executor.shutdownNow();
      executor.awaitTermination(executorShutdownTimeout, TimeUnit.SECONDS);
    }
  }

  public static void main(String[] args) throws InterruptedException {
    // The path to our example logistic regression model: update this path to load your own
    // models. In practice, models could be in a resources directory or on an accessible
    // filesystem. For our multi-threaded example, the model will be loaded by the main thread.
    String path = "my_models/lr_pipeline";
    LocalModel model = ModelFactory.loadModel(path);
    System.out.println(model);

    // Select output columns which the model should return.
    // Here, we output the new "prediction" and "probability" columns
    // and keep the existing "label" column.
    // The "probability" column contains a Vector of predicted probabilities for each class.
    String[] outputs = {"label", "prediction", "probability"};
    model.setOutputCols(outputs);

    final int numTasks = 129;

    // Create dataset
    final int featureSize = 13;
    List<TaskRecord> inputs = new FeatureGenerator(featureSize).newFeatures(numTasks);

    List<TaskRecord> resultsA = runTestWithParallelism(model, inputs, 4, 16);
    Collections.sort(resultsA);
    List<TaskRecord> resultsB = runTestWithParallelism(model, inputs, 8, 24);
    Collections.sort(resultsB);
    assert (resultsA.size() == numTasks && resultsB.size() == numTasks);
    for (int idx = 0; idx < numTasks; ++idx) {
      assert (resultsA.get(idx) == resultsB.get(idx));
    }
  }
}
