import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector

object NewsAnalysis {
  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("News Topic Modeling and Sentiment Analysis")
      .config("spark.master", "local[*]")
      .getOrCreate()

    import spark.implicits._

    // Load data from HDFS
    val hdfsPath = "hdfs://localhost:9000/input/data/newsdata.csv"
    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(hdfsPath)

    println(s"Total records loaded: ${rawData.count()}")
    rawData.printSchema()

    // Combine title and article for analysis, handle nulls
    val processedData = rawData
      .filter(col("title").isNotNull && col("article").isNotNull)
      .withColumn("text", concat_ws(" ", col("title"), col("article")))
      .withColumn("id", monotonically_increasing_id())
      .select("id", "text", "title", "date", "author", "section", "publication")

    processedData.cache()
    println(s"Processed records: ${processedData.count()}")

    // Text Preprocessing Pipeline
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("raw_tokens")
      .setPattern("\\W+")
      .setMinTokenLength(3)
    val remover = new StopWordsRemover()
      .setInputCol("raw_tokens")
      .setOutputCol("tokens")
    val hashingTF = new HashingTF()
      .setInputCol("tokens")
      .setOutputCol("raw_features")
      .setNumFeatures(10000)
    val idf = new IDF()
      .setInputCol("raw_features")
      .setOutputCol("features")
      .setMinDocFreq(5)
    val preprocessPipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf))

    val preprocessModel = preprocessPipeline.fit(processedData)
    val vectorizedData = preprocessModel.transform(processedData)
    vectorizedData.cache()

    // Topic Modeling with LDA
    val numTopics = 10
    val maxIterations = 50
    val lda = new LDA()
      .setK(numTopics)
      .setMaxIter(maxIterations)
      .setFeaturesCol("features")
      .setTopicDistributionCol("topicDistribution")
      .setSeed(42)
    val ldaModel = lda.fit(vectorizedData)

    // Vocabulary for topics
    val vocabArray = vectorizedData
      .select(explode(col("tokens")).as("word"))
      .groupBy("word").count()
      .orderBy(desc("count"))
      .limit(10000)
      .select("word")
      .as[String]
      .collect()

    // Display topics
    println("\n=== Discovered Topics ===")
    val topics = ldaModel.describeTopics(15)
    topics.collect().foreach { row =>
      val topicId = row.getInt(0)
      val termIndices = row.getSeq[Int](1)
      val termWeights = row.getSeq[Double](2)
      println(s"\nTopic $topicId:")
      termIndices.zip(termWeights).take(10).foreach { case (idx, weight) =>
        if (idx < vocabArray.length)
          println(f"  ${vocabArray(idx)}%-20s $weight%.4f")
      }
    }

    // Assign dominant topic to each document
    val getDominantTopic = udf((topicDist: Vector) => {
      topicDist.toArray.zipWithIndex.maxBy(_._1)._2
    })

    val topicResults = ldaModel.transform(vectorizedData)
    val documentsWithTopics = topicResults
      .withColumn("dominant_topic", getDominantTopic(col("topicDistribution")))
      .select("id", "title", "section", "publication", "date", "dominant_topic")

    // Simple rule-based sentiment analysis
    val sentimentUDF = udf((text: String) => {
      val positiveWords = Set("good", "great", "excellent", "amazing", "wonderful", "fantastic", "best", "happy", "love", "success", "win", "growth", "positive")
      val negativeWords = Set("bad", "terrible", "awful", "worst", "hate", "fail", "loss", "negative", "poor", "crisis", "death", "disaster", "war")
      val lowerText = text.toLowerCase()
      val words = lowerText.split("\\W+")
      val posCount = words.count(positiveWords.contains)
      val negCount = words.count(negativeWords.contains)
      if (posCount > negCount) "positive"
      else if (negCount > posCount) "negative"
      else "neutral"
    })

    val sentimentResults = processedData
      .withColumn("sentiment", sentimentUDF(col("text")))
      .select("id", "sentiment")  // FIXED: Only select id and sentiment to avoid duplicate columns

    // Combined Analysis - FIXED: No ambiguous column references now
    val finalResults = documentsWithTopics
      .join(sentimentResults, "id")
      .select(
        col("title"),
        col("section"),
        col("publication"),
        col("date"),
        col("dominant_topic"),
        col("sentiment")
      )

    // Topic Distribution by Section
    println("\n=== Topic Distribution by Section ===")
    finalResults
      .groupBy("section", "dominant_topic")
      .count()
      .orderBy(col("section"), desc("count"))
      .show(50)

    // Sentiment Distribution by Topic
    println("\n=== Sentiment Distribution by Topic ===")
    finalResults
      .groupBy("dominant_topic", "sentiment")
      .count()
      .orderBy(col("dominant_topic"), col("sentiment"))
      .show(50)

    // Sentiment Distribution by Publication
    println("\n=== Sentiment Distribution by Publication ===")
    finalResults
      .groupBy("publication", "sentiment")
      .count()
      .orderBy(col("publication"), desc("count"))
      .show(30)

    // Save Results to HDFS
    val outputPath = "hdfs://localhost:9000/user/hari/news_analysis_results"
    finalResults.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$outputPath/final_results")
    documentsWithTopics.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$outputPath/topic_assignments")
    sentimentResults.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$outputPath/sentiment_scores")
    ldaModel.save(s"$outputPath/lda_model")

    println(s"\n=== Results saved to HDFS: $outputPath ===")
    val ll = ldaModel.logLikelihood(vectorizedData)
    val lp = ldaModel.logPerplexity(vectorizedData)
    println(f"\nLog Likelihood: $ll%.2f")
    println(f"Log Perplexity: $lp%.2f")
    spark.stop()
  }
}
