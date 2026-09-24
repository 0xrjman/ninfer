#include "serve/openai_systemone.h"

#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void throw_invalid(const std::string& message) {
    ApiError error;
    error.status  = 422;
    error.type    = "validation_error";
    error.message = message;
    throw ApiException(std::move(error));
}

std::string normalize_value(const RequestJson& value) {
    if (value.is_null()) { return std::string(); }
    if (value.is_string()) { return value.get<std::string>(); }
    return value.dump();
}

const std::vector<ChoiceLabel>& choice_label_pool(
    const std::function<std::vector<ninfer::TokenId>(const std::string&)>& tokenize) {
    static std::vector<ChoiceLabel> pool;
    if (!pool.empty()) { return pool; }
    const char singles[]  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::vector<std::string> surfaces;
    for (const char c : singles) { surfaces.emplace_back(1, c); }
    for (const char a : alphabet) {
        for (const char b : alphabet) { surfaces.emplace_back(std::string{a, b}); }
    }
    std::unordered_set<ninfer::TokenId> used;
    for (const std::string& surface : surfaces) {
        const std::vector<ninfer::TokenId> ids = tokenize(surface);
        if (ids.size() != 1 || !used.insert(ids.front()).second) { continue; }
        pool.emplace_back(surface, ids.front());
        if (pool.size() == 255) { break; }
    }
    if (pool.size() < 255) {
        for (const char a : alphabet) {
            for (const char b : alphabet) {
                for (const char c : alphabet) {
                    if (pool.size() == 255) { break; }
                    const std::string surface{a, b, c};
                    const std::vector<ninfer::TokenId> ids = tokenize(surface);
                    if (ids.size() != 1 || !used.insert(ids.front()).second) { continue; }
                    pool.emplace_back(surface, ids.front());
                }
            }
        }
        if (pool.size() < 255) {
            pool.clear();
            throw std::runtime_error("could not collect 255 distinct single-token choice labels");
        }
    }
    return pool;
}

} // namespace

SystemOneRequest parse_systemone_request(const RequestJson& body) {
    SystemOneRequest request;
    request.model = body.value("model", std::string("ninfer-jev"));

    if (!body.contains("state") || body["state"].is_null()) { throw_invalid("state: required"); }
    request.state_text = normalize_value(body["state"]);

    if (!body.contains("questions") || !body["questions"].is_object() ||
        body["questions"].empty()) {
        throw_invalid("questions: needs a non-empty map of id -> question");
    }
    for (auto it = body["questions"].begin(); it != body["questions"].end(); ++it) {
        if (!it.value().is_object()) { throw_invalid("question " + it.key() + ": must be an object"); }
        const RequestJson& q = it.value();
        const std::string type = q.value("type", std::string());
        if (type != "choice" && type != "score" && type != "noul") {
            throw_invalid("question " + it.key() + ": unknown type " +
                          (q.contains("type") ? "\"" + type + "\"" : "<missing>"));
        }

        SystemOneQuestion question;
        question.id         = it.key();
        question.type       = type;
        question.instructions = q.contains("instructions") ? normalize_value(q["instructions"])
                                                           : std::string();

        if (type == "choice") {
            const auto& criteria = q["criteria"];
            if (!criteria.is_object() || criteria.empty()) {
                throw_invalid("question " + it.key() + ": choice criteria must map options to descriptions");
            }
            if (criteria.size() > 255) {
                throw_invalid("question " + it.key() + ": at most 255 choice options (one token each)");
            }
            for (auto cit = criteria.begin(); cit != criteria.end(); ++cit) {
                question.choice_options.emplace_back(cit.key(), normalize_value(cit.value()));
            }
        } else if (type == "score") {
            const auto& criteria = q["criteria"];
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw_invalid("question " + it.key() + ": score criteria must be an ordered list of 2-10 levels");
            }
            for (const auto& level : criteria) { question.score_levels.push_back(normalize_value(level)); }
        } else {
            const auto& criteria = q["criteria"];
            if (criteria.is_object()) {
                question.noul_true  = criteria.contains("true") ? normalize_value(criteria["true"]) : std::string();
                question.noul_false = criteria.contains("false") ? normalize_value(criteria["false"]) : std::string();
            }
        }
        request.questions.push_back(std::move(question));
    }
    return request;
}

std::string build_systemone_prompt(const std::string& state_text, const SystemOneQuestion& q,
                                   const std::vector<ChoiceLabel>& choice_pool) {
    std::string prompt =
        "Answer a single question about the state below. Consider the state and the question, "
        "then reply with exactly one answer label and nothing else.\n\n";
    prompt += "State:\n";
    prompt += state_text;
    prompt += "\n\n";
    prompt += "Question:\n";
    prompt += q.instructions;
    prompt += "\n";

    if (q.type == "choice") {
        prompt += "\nOptions:\n";
        for (std::size_t i = 0; i < q.choice_options.size(); ++i) {
            prompt += "  ";
            prompt += choice_pool[i].first;
            prompt += ": ";
            prompt += q.choice_options[i].first;
            if (!q.choice_options[i].second.empty()) {
                prompt += " (";
                prompt += q.choice_options[i].second;
                prompt += ")";
            }
            prompt += "\n";
        }
        prompt += "Reply with the exact label of the correct option:";
    } else if (q.type == "score") {
        prompt += "\nLevels, from lowest to highest:\n";
        for (std::size_t i = 0; i < q.score_levels.size(); ++i) {
            prompt += "  ";
            prompt += option_labels(q)[i];
            prompt += ": ";
            prompt += q.score_levels[i];
            prompt += "\n";
        }
        prompt += "Reply with a single number, the matching level:";
    } else {
        prompt += "\n";
        prompt += "  yes: ";
        prompt += q.noul_true;
        prompt += "\n";
        prompt += "  no: ";
        prompt += q.noul_false;
        prompt += "\n";
        prompt += "Reply with a single word, yes or no:";
    }
    return prompt;
}

std::vector<std::string> option_labels(const SystemOneQuestion& q) {
    std::vector<std::string> labels;
    if (q.type == "noul") {
        labels.push_back("yes");
        labels.push_back("no");
        return labels;
    }
    const std::size_t n = q.type == "choice" ? q.choice_options.size() : q.score_levels.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (q.type == "choice") {
            labels.push_back(std::string(1, static_cast<char>('A' + i)));
        } else if (i < 9) {
            labels.push_back(std::to_string(i + 1));
        } else {
            labels.push_back(std::string(1, static_cast<char>('A' + (i - 9))));
        }
    }
    return labels;
}

std::vector<double> normalize_logprobs(const std::vector<float>& logprobs) {
    std::vector<double> probs(logprobs.size());
    if (logprobs.empty()) { return probs; }
    float max_logprob = logprobs.front();
    for (const float value : logprobs) { max_logprob = std::max(max_logprob, value); }
    double total = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        probs[i] = std::exp(static_cast<double>(logprobs[i]) - static_cast<double>(max_logprob));
        total += probs[i];
    }
    for (double& value : probs) { value /= total; }
    return probs;
}

nlohmann::json make_systemone_answer(const SystemOneQuestion& q, const std::vector<double>& probs) {
    const std::size_t n = probs.size();
    double max_prob = 0.0;
    for (const double value : probs) { max_prob = std::max(max_prob, value); }
    const double confidence =
        n > 1 ? std::clamp((static_cast<double>(n) * max_prob - 1.0) / (static_cast<double>(n) - 1.0),
                           0.0, 1.0)
              : 1.0;

    Json answer;
    answer["type"] = q.type;
    if (q.type == "noul") {
        answer["noul"] = probs.empty() ? 0.0 : probs[0]; // P(yes)
        return answer;
    }
    if (q.type == "choice") {
        Json probabilities = Json::object();
        std::size_t best = 0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            probabilities[q.choice_options[i].first] = probs[i];
            if (probs[i] > probs[best]) { best = i; }
        }
        answer["choice"]      = q.choice_options[best].first;
        answer["probabilities"] = std::move(probabilities);
        answer["confidence"] = confidence;
        return answer;
    }
    // score
    Json probabilities = Json::object();
    Json legend = Json::object();
    double score = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        const std::string key = std::to_string(i);
        probabilities[key] = probs[i];
        legend[key]        = i < q.score_levels.size() ? q.score_levels[i] : std::string();
        score             += static_cast<double>(i) * probs[i];
    }
    answer["score"]         = score;
    answer["legend"]        = std::move(legend);
    answer["probabilities"] = std::move(probabilities);
    answer["confidence"]    = confidence;
    return answer;
}

void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    SystemOneRequest request;
    try {
        request = parse_systemone_request(parse_json_body(req));
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    Json answers = Json::object();
    int input_tokens = 0;
    try {
        const auto& choice_pool =
            choice_label_pool([this](const std::string& s) { return service_->tokenize_text(s); });
        for (const SystemOneQuestion& question : request.questions) {
            const std::string prompt =
                build_systemone_prompt(request.state_text, question, choice_pool);
            const std::vector<ninfer::TokenId> prefix = service_->tokenize_text(prompt);
            std::vector<ninfer::TokenId> candidates;
            if (question.type == "choice") {
                for (std::size_t i = 0; i < question.choice_options.size(); ++i) {
                    candidates.push_back(choice_pool[i].second);
                }
            } else {
                const std::vector<std::string> labels = option_labels(question);
                candidates.reserve(labels.size());
                for (const std::string& label : labels) {
                    const std::vector<ninfer::TokenId> token_ids = service_->tokenize_text(label);
                    candidates.push_back(token_ids.empty() ? 0 : token_ids.front());
                }
            }
            const std::vector<float> logprobs =
                service_->score_candidates(prefix, std::move(candidates));
            input_tokens += static_cast<int>(prefix.size());
            answers[question.id] = make_systemone_answer(question, normalize_logprobs(logprobs));
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    Json body{
        {"model", "ninfer-jev"},
        {"answers", std::move(answers)},
        {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}},
    };
    try {
        set_owned_json_content(res, body.dump(), {});
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
