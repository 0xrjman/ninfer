#pragma once

// Jev / "System One" wire adapter for POST /v1/systemone. A request carries one state plus a set
// of typed questions (choice / score / noul). Each question is answered by scoring the model's
// next-token distribution over the option's single candidate tokens; no text is generated.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

struct SystemOneQuestion {
    std::string id;
    std::string type; // "choice" | "score" | "noul"
    std::string instructions;
    std::vector<std::pair<std::string, std::string>> choice_options; // (name, description)
    std::vector<std::string> score_levels;                           // ordered level descriptions
    std::string noul_true;
    std::string noul_false;
};

struct SystemOneRequest {
    std::string model;
    std::string state_text; // normalized state (string as-is, object/array as JSON)
    std::vector<SystemOneQuestion> questions; // request insertion order
};

SystemOneRequest parse_systemone_request(const RequestJson& body);

// One question's prompt: the state plus that question's criteria, ending where the model's next
// token is the answer label.
std::string build_systemone_prompt(const std::string& state_text, const SystemOneQuestion& q);

// Candidate tokens at the single answer position, in the same order as the probabilities output
// below: choice -> "A","B",... ; score -> "1","2",... ; noul -> "yes","no".
std::vector<std::string> option_labels(const SystemOneQuestion& q);

// Turns raw per-candidate log-probs into a sum-to-1 distribution (softmax), in order.
std::vector<double> normalize_logprobs(const std::vector<float>& logprobs);

// Shapes one answer exactly per the Jev contract from the normalized distribution.
nlohmann::json make_systemone_answer(const SystemOneQuestion& q, const std::vector<double>& probs);

} // namespace ninfer::serve
