package Mustafin

import org.apache.log4j.BasicConfigurator
import org.apache.log4j.varia.NullAppender
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.DecimalType
import org.jfree.chart._
import org.jfree.data.category.DefaultCategoryDataset
import org.jfree.chart.ChartFactory
import org.jfree.data.general.DefaultPieDataset

object Analyzing{
  val PATH: String = "src/main/data"
  val NODES: Int = 3

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
      .option("quote","\"")
      .load(s"$PATH/var_03.csv")
      .withColumnRenamed("Average Life Expectancy (Years)","Average")
      .withColumnRenamed("Age-adjusted Death Rate","Death_Rate")
      .withColumn("Average", col("Average").cast("float"))
      .withColumn("Death_Rate", col("Death_Rate").cast("float"))

    dataframe.show(20)

    println("Количество записей в исходном датасете")
    println(dataframe.count)

    val filteredDF: DataFrame = dataframe.filter(col("Average").isNotNull)

    filteredDF.show(20)

    println("Количество записей (исключены Null строки)")
    println(filteredDF.count)
    //endregion

    //region Description

    val description: DataFrame = filteredDF.describe("Average", "Death_Rate")
    description.show()

    val temp1: Array[String] = description.rdd.map(row=>row.getString(1)).collect
    val stdAverage: Float = temp1(2).toFloat
    val meanAverage: Float = temp1(1).toFloat

    val temp2: Array[String] = description.rdd.map(row=>row.getString(2)).collect
    val stdDeath_Rate: Float = temp2(2).toFloat
    val meanDeath_Rate: Float = temp2(1).toFloat
    //endregion

    //region Year
    println("Среднее по годам")
    val meanPerYear: DataFrame = filteredDF.groupBy("Year")
      .mean( "Death_Rate", "Average")
      .withColumn("avg(Death_Rate)", col("avg(Death_Rate)")
      .cast("float"))
      .withColumn("avg(Average)", col("avg(Average)")
      .cast("float"))
      .withColumn("Year", col("Year")
      .cast("int"))
      .sort("Year")

    meanPerYear.show(30)

    meanPerYear.agg(min("avg(Death_Rate)"), max("avg(Death_Rate)")).show()
    meanPerYear.agg(min("avg(Average)"), max("avg(Average)")).show()

    val data = getDataFromDF(meanPerYear, "Year", "avg(Death_Rate)")
    val dataSet = new DefaultCategoryDataset()
    data.foreach(x=> dataSet.setValue(x._2, "Средняя смертность", x._1))

    val frame = new ChartFrame(
      "Средняя смертность",
      ChartFactory.createLineChart(
        "Средняя смертность",
        "Год",
        "Смертей на 100 000",
        dataSet,
        org.jfree.chart.plot.PlotOrientation.VERTICAL,
        false,false,false
      )
    )
    frame.pack()
    frame.setVisible(true)

    val data2 = getDataFromDF(meanPerYear, "Year", "avg(Average)")
    val dataSet2 = new DefaultCategoryDataset()
    data2.foreach(x=> dataSet2.setValue(x._2, "Средняя продолжительность жизни", x._1))

    val frame2 = new ChartFrame(
      "Средняя продолжительность жизни",
      ChartFactory.createLineChart(
        "Средняя продолжительность жизни",
        "Год",
        "Лет",
        dataSet2,
        org.jfree.chart.plot.PlotOrientation.VERTICAL,
        false,false,false
      )
    )
    frame2.pack()
    frame2.setVisible(true)

    //endregion

    //region mean

    println("Средняя продолжительность жизни по рассе")
    val RaceAverageMean: DataFrame = filteredDF
      .groupBy("Race")
      .mean("Average")
      .withColumn("avg(Average)", col("avg(Average)")
      .cast(DecimalType(3, 1)))
      .sort("Race")
    RaceAverageMean.show()

    println("Средняя продолжительность жизни по полу")
    val SexAverageMean: DataFrame = filteredDF
      .groupBy("Sex")
      .mean("Average")
      .withColumn("avg(Average)", col("avg(Average)")
      .cast(DecimalType(3, 1)))
      .sort("Sex")
    SexAverageMean.show()

    println("Средняя продолжительность жизни по рассе и полу")
    val RaceSexAverageMean: DataFrame = filteredDF
      .groupBy("Race", "Sex")
      .mean("Average")
      .withColumn("avg(Average)", col("avg(Average)")
      .cast("float"))
      .sort("Race", "Sex")
    RaceSexAverageMean.show()

    getBarChart("Sex/Race Average", "sex", getDataToBarChart(RaceSexAverageMean, "Race", "Sex", "avg(Average)"))

    println("Среднее количество смертей на 100 000 человек распределенное по рассе")
    val RaceDeath_RateMean: DataFrame = filteredDF
      .groupBy("Race")
      .mean("Death_Rate")
      .withColumn("avg(Death_Rate)", col("avg(Death_Rate)")
      .cast(DecimalType(5, 1)))
      .sort("Race")
    RaceDeath_RateMean.show()

    println("Среднее количество смертей на 100 000 человек распределенное по полу")
    val SexDeath_RateMean: DataFrame = filteredDF
      .groupBy("Sex")
      .mean("Death_Rate")
      .withColumn("avg(Death_Rate)", col("avg(Death_Rate)")
      .cast(DecimalType(5, 1)))
      .sort("Sex")
    SexDeath_RateMean.show()

    println("Среднее количество смертей на 100 000 человек распределенное по рассе и полу")
    val RaceSexDeath_RateMean: DataFrame = filteredDF
      .groupBy("Race", "Sex")
      .mean("Death_Rate")
      .withColumn("avg(Death_Rate)", col("avg(Death_Rate)")
      .cast("float"))
      .sort("Race", "Sex")
    RaceSexDeath_RateMean.show()

    getBarChart("Sex/Race Death Rate", "sex", getDataToBarChart(RaceSexDeath_RateMean, "Race", "Sex", "avg(Death_Rate)"))
    //endregion

    //region standart deviation

    println("Стандартное отклонение продолжительности жизни по рассе")
    val RaceAverageSTD: DataFrame = filteredDF
      .groupBy("Race")
      .agg(stddev("Average"))
      .withColumnRenamed("stddev_samp(Average)","STD(Average)")
      .withColumn("STD(Average)", col("STD(Average)")
      .cast(DecimalType(3, 1)))
      .sort("Race")
    RaceAverageSTD.show()

    println("Стандартное отклонение продолжительности жизни по полу")
    val SexAverageSTD: DataFrame = filteredDF
      .groupBy("Sex")
      .agg(stddev("Average"))
      .withColumnRenamed("stddev_samp(Average)","STD(Average)")
      .withColumn("STD(Average)", col("STD(Average)")
      .cast(DecimalType(3, 1)))
      .sort("Sex")
    SexAverageSTD.show()

    println("Стандартное отклонение продолжительности жизни по рассе и полу")
    val RaceSexAverageSTD: DataFrame = filteredDF
      .groupBy("Race", "Sex")
      .agg(stddev("Average"))
      .withColumnRenamed("stddev_samp(Average)","STD(Average)")
      .withColumn("STD(Average)", col("STD(Average)")
      .cast(DecimalType(3, 1)))
      .sort("Race", "Sex")
    RaceSexAverageSTD.show()

    println("Стандартное отклонение количества смертей на 100 000 человек распределенное по рассе")
    val RaceDeath_RateSTD: DataFrame = filteredDF
      .groupBy("Race")
      .agg(stddev("Death_Rate"))
      .withColumnRenamed("stddev_samp(Death_Rate)","STD(Death_Rate)")
      .withColumn("STD(Death_Rate)", col("STD(Death_Rate)")
      .cast(DecimalType(4, 1)))
      .sort("Race")
    RaceDeath_RateSTD.show()

    println("Стандартное отклонение количества смертей на 100 000 человек распределенное по полу")
    val SexDeath_RateSTD: DataFrame = filteredDF
      .groupBy("Sex")
      .agg(stddev("Death_Rate"))
      .withColumnRenamed("stddev_samp(Death_Rate)","STD(Death_Rate)")
      .withColumn("STD(Death_Rate)", col("STD(Death_Rate)")
      .cast(DecimalType(4, 1)))
      .sort("Sex")
    SexDeath_RateSTD.show()

    println("Стандартное отклонение количества смертей на 100 000 человек распределенное по рассе и полу")
    val RaceSexDeath_RateSTD: DataFrame = filteredDF
      .groupBy("Race", "Sex")
      .agg(stddev("Death_Rate"))
      .withColumnRenamed("stddev_samp(Death_Rate)","STD(Death_Rate)")
      .withColumn("STD(Death_Rate)", col("STD(Death_Rate)")
      .cast(DecimalType(4, 1)))
      .sort("Race", "Sex")
    RaceSexDeath_RateSTD.show()
    //endregion

    //region Dataset with small Life expectancy

    val shortLive: DataFrame = filteredDF
      .filter(col("Average") < meanAverage - stdAverage * 2
        || col("Death_Rate") < meanDeath_Rate - stdDeath_Rate * 2)
      .sort(col("Year"))

    println("Характеристики датасета с низкой продолжительностью жизни")
    shortLive.describe("Average", "Death_Rate").show()

    println("Датасет с низкой продолжительностью жизни")
    shortLive.show(shortLive.count().toInt)

    println("Процент высокой смертности по Рассе")
    val df1: DataFrame = shortLive
      .groupBy("Race")
      .agg(count("Race").alias("Count"))
      .withColumn("Percent", col("Count") /  sum("Count").over() * 100)
      .withColumn("Percent", col("Percent")
      .cast("float"))
      .sort(col("Count").desc)

    df1.show()

    getPieChart("Высокая смертность по Рассе", getDataPie(df1, "Race", "Percent"))

    println("Процент высокой смертности по Полу")
    val df2: DataFrame = shortLive
      .groupBy("Sex")
      .agg(count("Sex").alias("Count"))
      .withColumn("Percent", col("Count") /  sum("Count").over() * 100)
      .withColumn("Percent", col("Percent")
      .cast("float"))
      .sort(col("Count").desc)

    df2.show()

    getPieChart("Высокая смертность по Рассе", getDataPie(df2, "Sex", "Percent"))
    //endregion
  }

  def getDataFromDF(df: DataFrame, col1: String, col2: String): (Array[(Int, Float)]) =
  {
    val localRows = df.take(df.count().toInt)
    val l: Array[Int] = localRows.map(_.getAs[Int](col1))
    val c: Array[Float] = localRows.map(_.getAs[Float](col2))
    (l,c).zipped.toArray
  }

  def getDataToBarChart(df: DataFrame, x : String, y : String, stringCount : String): Array[(String, String, Float)] =
  {
    val localRows = df.take(df.count().toInt)
    val xRow: Array[String] = localRows.map(_.getAs[String](x))
    val yRow: Array[String] = localRows.map(_.getAs[String](y))
    val count: Array[Float] = localRows.map(_.getAs[Float](stringCount))

    (xRow, yRow, count).zipped.toArray
  }

  def getBarChart(x : String, y : String, data: Array[(String, String, Float)]): Unit =
  {
    var dataSet = new DefaultCategoryDataset()
    data.foreach(x=> dataSet.setValue(x._3,x._2,x._1))

    val chart = new ChartFrame(
      "",
      ChartFactory.createBarChart(
        "",
        x,
        y,
        dataSet
      ))
    chart.pack()
    chart.setVisible(true)
  }

  def getDataPie(df: DataFrame, labelsRow : String, valuesRow : String): Array[(String, Float)] =
  {
    val localRows = df.take(df.count().toInt)
    val labels: Array[String] = localRows.map(_.getAs[String](labelsRow))
    val values: Array[Float] = localRows.map(_.getAs[Float](valuesRow))
    (labels, values).zipped.toArray
  }

  def getPieChart(title : String, data: Array[(String, Float)]): Unit =
  {
    var dataSet = new DefaultPieDataset()
    data.foreach(x=> dataSet.setValue(x._1,x._2))

    val chart = new ChartFrame(
      title,
      ChartFactory.createPieChart(
        title,
        dataSet
      ))
    chart.pack()
    chart.setVisible(true)
  }
}
