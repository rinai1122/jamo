import collections
import json
from jamo import h2j, j2hcj
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_jamo_sequence(text):
    return j2hcj(h2j(text))

def build_frequency_table(text):
    jamos = get_jamo_sequence(text)
    unigrams = collections.Counter(jamos)
    bigrams = collections.Counter(zip(jamos, jamos[1:]))
    trigrams = collections.Counter(zip(jamos, jamos[1:], jamos[2:]))
    
    def normalize(counter):
        total = sum(counter.values())
        return {("".join(k) if isinstance(k, tuple) else k): (v / total) for k, v in counter.items()}

    return {
        "unigrams": normalize(unigrams),
        "bigrams": normalize(bigrams),
        "trigrams": normalize(trigrams)
    }

if __name__ == "__main__":
    # Much larger corpus simulation
    corpus = [
        "안녕하세요 반갑습니다", "오늘 날씨가 참 좋네요", "맛있는 점심 드셨나요",
        "한국어 공부는 재미있어요", "세종대왕이 한글을 만드셨습니다", "독도는 우리 땅입니다",
        "대한민국의 주권은 국민에게 있습니다", "민주공화국 대한민국", "나랏말싸미 듕귁에 달아",
        "가는 말이 고와야 오는 말이 곱다", "원숭이도 나무에서 떨어집니다", "아는 것이 힘이다",
        "백지장도 맞들면 낫습니다", "소 잃고 외양간 고치기", "호랑이에게 물려가도 정신만 차리면 산다",
        "하늘이 무너져도 솟아날 구멍은 있다", "시작이 반이다", "천 리 길도 한 걸음부터",
        "고생 끝에 낙이 온다", "실패는 성공의 어머니", "시간은 금이다", "건강이 최고다"
    ] * 50 # Repeat to boost stats
    
    training_text = "\n".join(corpus)
    freq_data = build_frequency_table(training_text)
    
    # Add a small word list for dictionary fitness
    words = set()
    for line in corpus:
        words.update(line.split())
    freq_data["dictionary"] = list(words)
    
    with open("freq_table.json", "w", encoding="utf-8") as f:
        json.dump(freq_data, f, ensure_ascii=False, indent=2)
    
    print("Enhanced frequency table and dictionary built.")
