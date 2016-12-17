package com.tikal.spark

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import java.io.File

import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.sql.functions._


object App {

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().appName("test").master("local[*]").getOrCreate()
        val sc = spark.sparkContext
        val sqlContext = spark.sqlContext
        import sqlContext.sql
        import sqlContext.implicits._

        if(args.length != 2)
            throw new RuntimeException("Wrong number of parameters. Usage is commad-to-run <movies-file.dat> <ratings-file.dat>")
       sc.textFile(args(0)).map { line =>
            val fields = line.split("::")
            // format: (movieId, movieName)
            (fields(0).toInt, fields(1))
        }.toDF("movieId","movieName").createOrReplaceTempView("movies")


        val ratings: RDD[Rating] = sc.textFile(args(1)).map { line =>
            val fields = line.split("::")
            // format:  Rating(userId, movieId, rating)
            Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
        }

        // Build the model using the training set
        val split: Array[RDD[Rating]] = ratings.randomSplit(Array(0.8, 0.2))
        val trainingRDD = split(0).cache()
        val model = ALS.train(trainingRDD,10, 10)

        // recommend top 5 products to user 4169
        val products: Array[Rating] = model.recommendProducts(4169, 5)
        sc.parallelize(products).toDF().drop("user").show(false)




        val testRDD = split(1).cache()
        testRDD.map(x=>(x.user,x.product,x.rating)).toDF("user","product","actual_rating").createOrReplaceTempView("test_tbl")

        // Show prediction table
        val testUserMovieRDD = testRDD.map(x=>(x.user,x.product))
        val predictionsTestRDD: RDD[(Int, Int, Double)] = model.predict(testUserMovieRDD).map(r=>(r.user,r.product,r.rating))
        predictionsTestRDD.toDF("user","product","predicated_rating").createOrReplaceTempView("prediction_tbl")
        sql(
            """
              select p.user, movieId, movieName, predicated_rating , actual_rating
              from prediction_tbl p
              join test_tbl t on p.user=t.user and p.product=t.product
              join movies m on m.movieId = p.product
            """).createOrReplaceTempView("ratings_tbl")

        sql("select * from ratings_tbl").show(false)

        sql("""
              select ((predicated_rating - actual_rating)*(predicated_rating - actual_rating)) as sqr_diff
              from ratings_tbl
            """).select(avg($"sqr_diff")).show(false)

    }
}
