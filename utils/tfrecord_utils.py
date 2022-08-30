from typing import List
import tensorflow as tf


def bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value: List[bytes]):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value: List[int]):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value: List[float]):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))