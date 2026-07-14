from enum import Enum, StrEnum

class AnalysisProps(StrEnum):
    CLASS = "class"
    VAL = "valence"
    ACT = "activation"
    DOM = "dominance"
    INT = "intensity_class"
    TIME = "time"
    IDX = "keyframe"

class EmoTask(Enum):
    EMOCLASS = "emo_class"
    VAD = "emo_vad"
    INTCLASS = "emo_intensity_class"

class Emotions(StrEnum):
    NEU = "neutral"
    HAP = "happiness"
    SAD = "sadness"
    ANG = "anger"
    FEA = "fear"
    DIS = "disgust"
    SUR = "surprise"
    CON = "contempt"

class IntensityLevels(StrEnum):
    NOR = "normal"
    SNG = "strong"

class MSPVADNormalization():
    @staticmethod
    def normalize(valence):
        normalized = (valence - 1) / (7 - 1)
        return normalized
