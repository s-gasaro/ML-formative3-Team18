import csv

# keywords I'm testing
positive_keywords = ["wonderful", "excellent", "brilliant"]
negative_keywords = ["boring", "awful", "waste"]
all_keywords = positive_keywords + negative_keywords


def load_and_count(filepath):
    total_positive = 0
    total_negative = 0
    keyword_in_positive = {word: 0 for word in all_keywords}
    keyword_in_negative = {word: 0 for word in all_keywords}

    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            review_text = row[0].lower()
            sentiment = row[1]

            if sentiment == "positive":
                total_positive += 1
            else:
                total_negative += 1

            for word in all_keywords:
                if word in review_text:
                    if sentiment == "positive":
                        keyword_in_positive[word] += 1
                    else:
                        keyword_in_negative[word] += 1

    return total_positive, total_negative, keyword_in_positive, keyword_in_negative


def compute_bayes_table(total_positive, total_negative, keyword_in_positive, keyword_in_negative):
    total_reviews = total_positive + total_negative
    prior_positive = total_positive / total_reviews  # P(Positive)

    results = {}
    for word in all_keywords:
        pos_count = keyword_in_positive[word]
        neg_count = keyword_in_negative[word]
        keyword_total = pos_count + neg_count

        likelihood = pos_count / total_positive          # P(keyword | Positive)
        marginal = keyword_total / total_reviews          # P(keyword)
        posterior = (likelihood * prior_positive) / marginal if marginal != 0 else 0  # P(Positive | keyword)

        results[word] = {
            "prior": prior_positive,
            "likelihood": likelihood,
            "marginal": marginal,
            "posterior": posterior,
            "pos_count": pos_count,
            "neg_count": neg_count,
        }

    return results


def print_table(results):
    print(f"{'Keyword':12s} | {'Prior':8s} | {'Likelihood':12s} | {'Marginal':10s} | {'Posterior':10s}")
    for word in all_keywords:
        r = results[word]
        print(f"{word:12s} | {r['prior']:.4f}   | {r['likelihood']:.4f}       | {r['marginal']:.4f}     | {r['posterior']:.4f}")
    print()


if __name__ == "__main__":
    filepath = "IMDB Dataset.csv"
    total_pos, total_neg, kw_pos, kw_neg = load_and_count(filepath)

    print(f"Total reviews: {total_pos + total_neg}")
    print(f"Positive: {total_pos}, Negative: {total_neg}\n")

    results = compute_bayes_table(total_pos, total_neg, kw_pos, kw_neg)
    print_table(results)

    print("Word counts:")
    for word in all_keywords:
        r = results[word]
        print(f"  {word}: {r['pos_count']} positive reviews, {r['neg_count']} negative reviews")