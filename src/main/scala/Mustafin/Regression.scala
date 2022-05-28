package Mustafin

import org.apache.log4j.BasicConfigurator
import org.apache.log4j.varia.NullAppender
import org.apache.spark.ml.feature.{Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

object Regression {
  val PATH: String = "src/main/data"
  val NODES: Int = 3
  val SEED: Int = 42

  def main(args: Array[String]): Unit = {
    //region Base
    BasicConfigurator.configure(new NullAppender)

    val spark: SparkSession = SparkSession.builder()
      .appName("Lab3")
      .master(s"local[$NODES]")
      .getOrCreate

    val dataframe: DataFrame = spark
      .read
      .format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("quote", "\"")
      .load(s"$PATH/var_03.csv")
      .withColumnRenamed("Average Life Expectancy (Years)", "Average")
      .withColumnRenamed("Age-adjusted Death Rate", "Death_Rate")
      .withColumnRenamed("Average", "label")
      .withColumn("label", col("label").cast("float"))
      .withColumn("Death_Rate", col("Death_Rate").cast("float"))
      .withColumn("Year", col("Year").cast("float"))

    val filteredDF: DataFrame = dataframe.filter(col("label").isNotNull)

    val tempDF2: DataFrame = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("indexSex")
      .fit(filteredDF)
      .transform(filteredDF)

    val tempDF: DataFrame = new StringIndexer()
      .setInputCol("Race")
      .setOutputCol("indexRace")
      .fit(tempDF2)
      .transform(tempDF2)

    tempDF.show()

    val cols: Array[String] = Array("Year", "Death_Rate", "indexSex", "indexRace")

    // Indexing some columns for using as features
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val workDF: DataFrame = assembler.transform(tempDF)

    val normDF: DataFrame = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2.0)
      .transform(workDF)

    workDF.show(false)
    normDF.show(false)

    val Array(training, test) = normDF.randomSplit(Array(0.7, 0.3), SEED)

    println(s"dataframe count: ${filteredDF.count()}")
    println(s"training count: ${training.count()}")
    println(s"test count: ${test.count()}")

    val lr = new LinearRegression()
//      .setFeaturesCol("features")
      .setFeaturesCol("normFeatures")
      .setMaxIter(20)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(training)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val prediction: DataFrame = lrModel
      .transform(test)

    prediction
      .select("Year", "Death_Rate", "indexSex", "indexRace", "label", "prediction")
      .show()
  }
}
