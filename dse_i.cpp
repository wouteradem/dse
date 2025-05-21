#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <regex>
#include <numeric>
#include <cmath>
#include <complex>
#include <fstream>

struct Term {
    std::vector<int> indices;
    std::complex<double> coefficient;

    Term(std::vector<int> idx, std::complex<double> coefficient) : indices(idx), coefficient(coefficient) {}

    friend std::ostream& operator<<(std::ostream& os, const Term& t) {
	    os << "(";

	    for (size_t i = 0; i < t.indices.size(); ++i) {
	        os << t.indices[i];
	        if (i < t.indices.size() - 1) os << ",";
	    }

	    os << ";(" << std::fixed << std::setprecision(6)
	       << std::real(t.coefficient) << "," << std::imag(t.coefficient) << "))";

	    return os;
	}

	std::string to_string() const {
	    std::ostringstream oss;
	    oss << "(";

	    for (size_t i = 0; i < indices.size(); ++i) {
	        oss << indices[i];
	        if (i < indices.size() - 1) oss << ",";
	    }

	    oss << ";(" << std::fixed << std::setprecision(6)
	        << std::real(coefficient) << "," << std::imag(coefficient) << "))";

	    return oss.str();
	}
};


struct Equation {
    std::vector<Term> terms;

    friend std::ostream& operator<<(std::ostream& os, const Equation& eq) {
	    os << "{";
	    for (size_t i = 0; i < eq.terms.size(); ++i) {
	        os << eq.terms[i];
	        if (i < eq.terms.size() - 1) os << ",";
	    }
	    os << "}";
	    return os;
	}

	std::string to_string() const {
	    std::ostringstream oss;
	    oss << "{";
	    for (size_t i = 0; i < terms.size(); ++i) {
	        oss << terms[i];
	        if (i < terms.size() - 1) oss << ",";
	    }
	    oss << "}";
	    return oss.str();
	}
};
            

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \n\r\t");
    size_t end = s.find_last_not_of(" \n\r\t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}                                               

std::string substitute(std::string equation, std::map<int, std::string>& equations, int index) {
    std::regex tuple_regex(R"(\(([^;]+);\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)\))");
    std::regex number_regex(R"(\d+)");
    std::smatch match;

    std::string m_equation = "{";
    std::string::const_iterator search_start(equation.cbegin());

    size_t total_matches = std::distance(
        std::sregex_iterator(equation.begin(), equation.end(), tuple_regex),
        std::sregex_iterator());
    size_t current_match = 0;

    while (std::regex_search(search_start, equation.cend(), match, tuple_regex)) {
        std::string before_semicolon = match[1].str();
        std::string real_part = match[2].str();
        std::string imag_part = match[3].str();
        std::string after_semicolon = ";(" + real_part + "," + imag_part + ")";

        m_equation += "(";
        std::smatch num_match;
        std::string::const_iterator num_search_start(before_semicolon.cbegin());

        bool first_number = true;
        while (std::regex_search(num_search_start, before_semicolon.cend(), num_match, number_regex)) {
            int num = std::stoi(num_match[0]);

			std::cout << "Checking index: " << index << std::endl;
			for (int num = 0; num < 6; ++num) {
			    if (equations.count(num)) {
			        std::cout << "equations[" << num << "] = " << equations[num] << std::endl;
			    }
			}

            std::string replacement;
            if (num > index && equations.count(num) > 0) {
                std::cout << "Substituting " << num << " into index " << index
                          << " with {" << equations[num] << "}" << std::endl;
                replacement = "{" + equations[num] + "}";
            } else {
                replacement = num_match.str();
            }

            if (!first_number) m_equation += ",";
            first_number = false;

            m_equation += replacement;
            num_search_start = num_match.suffix().first;
        }

        m_equation += after_semicolon;
        if (++current_match < total_matches) m_equation += ",";

        search_start = match.suffix().first;
    }

    m_equation += "}";
    return m_equation;
}



long long binomial(int n, int k) {
    if (k > n - k) k = n - k;
    long long res = 1;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    
	return res;
}

void merge_terms(Equation &eq) {
    std::map<std::vector<int>, std::complex<double>> merged_terms;  // Map to sum coefficients of identical terms

    // Process each term
    for (const auto &term : eq.terms) {
        std::vector<int> sorted_indices = term.indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());  // Ensure consistent order

        // Add coefficient to existing term or insert a new one
        merged_terms[sorted_indices] += term.coefficient;
    }

    // Clear existing terms and replace with merged ones
    eq.terms.clear();
    for (const auto &entry : merged_terms) {
        eq.terms.emplace_back(entry.first, entry.second);
    }
}

std::vector<Equation> compute_i_equations(int m_max) {
	std::vector<Equation> equations;

	for (int m = 0; m < m_max; m++) {
		Equation eq;

		eq.terms.emplace_back(std::vector<int>{m + 2}, std::complex<double>(1.0, 0.0));
		for (int k = 0; k <= m; k++) {
			//std::cout << "in k loop " << std::endl;
			std::complex<double> coeff(static_cast<double>(binomial(m, k)), 0.0);
			eq.terms.emplace_back(std::vector<int>{k + 1, m - k + 1}, coeff);
		}

        merge_terms(eq);

		if (m == 1) {
			eq.terms.emplace_back(std::vector<int>{0, 0}, std::complex<double>(0.0, 1.0));
		}
		equations.push_back(eq);
	}

	return equations;	
}

Equation parse_equation(const std::string& input);

// Helper function.
size_t find_top_level_semicolon(const std::string& s) {
    int paren_depth = 0, brace_depth = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '(') paren_depth++;
        else if (c == ')') paren_depth--;
        else if (c == '{') brace_depth++;
        else if (c == '}') brace_depth--;
        else if (c == ';' && paren_depth == 0 && brace_depth == 0) {
            return i;
        }
    }
    throw std::runtime_error("Top-level semicolon not found in string: " + s);
}


Equation expand_nested(const std::vector<int>& outer_indices, std::complex<double> scalar, const Equation& nested) {
    Equation result;
    for (const auto& term : nested.terms) {
        std::vector<int> indices = outer_indices;
        indices.insert(indices.end(), term.indices.begin(), term.indices.end());
        result.terms.emplace_back(indices, scalar * term.coefficient);
    }
    return result;
}

std::vector<std::string> split_top_level(const std::string& s) {
    std::vector<std::string> parts;
    std::string current;
    int paren_depth = 0;
    int brace_depth = 0;

    for (char c : s) {
        if (c == '(') paren_depth++;
        if (c == ')') paren_depth--;
        if (c == '{') brace_depth++;
        if (c == '}') brace_depth--;

        if (c == ',' && paren_depth == 0 && brace_depth == 0) {
            parts.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }

    if (!current.empty()) {
        parts.push_back(trim(current));
    }

    return parts;
}

void parse_and_expand_term(const std::string& s, Equation& result) {
    if (s.front() != '(' || s.back() != ')')
        throw std::runtime_error("Malformed term: " + s);

    std::string inner = s.substr(1, s.size() - 2);  // remove outer parentheses
    size_t semi = find_top_level_semicolon(inner);
    std::string left = trim(inner.substr(0, semi));

    // --- Start robust complex parsing ---
    size_t coeff_start = semi + 1;
    if (inner[coeff_start] != '(')
        throw std::runtime_error("Expected '(' at complex coefficient start: " + inner.substr(coeff_start));

    // Find matching closing parenthesis for the complex number
    int depth = 0;
    size_t coeff_end = std::string::npos;
    for (size_t i = coeff_start; i < inner.size(); ++i) {
        if (inner[i] == '(') depth++;
        else if (inner[i] == ')') {
            depth--;
            if (depth == 0) {
                coeff_end = i;
                break;
            }
        }
    }

    if (coeff_end == std::string::npos)
        throw std::runtime_error("Could not find closing parenthesis for complex coefficient: " + inner);

    std::string coeff_str = inner.substr(coeff_start + 1, coeff_end - coeff_start - 1); // without parentheses

    size_t comma = coeff_str.find(',');
    if (comma == std::string::npos)
        throw std::runtime_error("Malformed complex pair: " + coeff_str);

    double real = std::stod(coeff_str.substr(0, comma));
    double imag = std::stod(coeff_str.substr(comma + 1));
    std::complex<double> coeff(real, imag);
    // --- End robust complex parsing ---

    std::vector<std::string> parts = split_top_level(left);
    std::vector<int> indices;
    bool nested_found = false;
    Equation nested_accum;

    for (const std::string& part : parts) {
        if (part.front() == '{') {
            Equation subeq = parse_equation(part);
            if (!nested_found) {
                nested_accum = subeq;
                nested_found = true;
            } else {
                Equation next;
                for (const auto& t1 : nested_accum.terms) {
                    for (const auto& t2 : subeq.terms) {
                        std::vector<int> merged = t1.indices;
                        merged.insert(merged.end(), t2.indices.begin(), t2.indices.end());
                        next.terms.emplace_back(merged, t1.coefficient * t2.coefficient);
                    }
                }
                nested_accum = next;
            }
        } else {
            indices.push_back(std::stoi(part));
        }
    }

    if (nested_found) {
        Equation expanded = expand_nested(indices, coeff, nested_accum);
        result.terms.insert(result.terms.end(), expanded.terms.begin(), expanded.terms.end());
    } else {
        result.terms.emplace_back(indices, coeff);
    }
}


void solve_polynomial(const Equation& parsed, const std::string& output_filename = "roots.csv") {
    std::cout << "Roots of the polynomial:\n";

    const int degree = 20; // ADJUST THIS IF NEEDED!
    std::vector<std::complex<double>> coeffs(degree + 1, std::complex<double>(0.0, 0.0));

    for (const auto& term : parsed.terms) {
        int x_pow = 0;
        for (int idx : term.indices) {
            if (idx == 1) x_pow++;  // Solve for unknown G_1
        }
        if (x_pow <= degree) {
            coeffs[x_pow] += term.coefficient;
        }
    }

    int deg = 0;
	for (int i = degree; i >= 0; --i) {
	    if (std::abs(coeffs[i]) > 1e-14) {
	        deg = i;
	        break;
	    }
	}

    if (deg == 0) {
        std::cout << "Polynomial is constant.\n";
        return;
    }

	std::complex<double> lead = coeffs[deg];
	if (std::abs(lead) > 1e-14) {
	    for (int i = 0; i <= deg; ++i) {
	        coeffs[i] /= lead;
	    }
	}

    std::vector<std::complex<double>> roots(deg);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < deg; ++i) {
        double angle = 2 * pi * i / deg;
        roots[i] = std::polar(1.0, angle);
    }

    const int max_iter = 1000;
    const double tol = 1e-12;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int i = 0; i < deg; ++i) {
            std::complex<double> prod = 1.0;
            for (int j = 0; j < deg; ++j) {
                if (i != j) prod *= (roots[i] - roots[j]);
            }

            std::complex<double> f = 0.0;
            for (int k = deg; k >= 0; --k) {
                f = f * roots[i] + coeffs[k];
            }

            std::complex<double> delta = f / prod;
            roots[i] -= delta;
            if (std::abs(delta) > tol) converged = false;
        }
        if (converged) break;
    }

    std::ofstream file(output_filename, std::ios::app);
    if (!file) {
        std::cerr << "Error: Unable to open file " << output_filename << "\n";
        return;
    }

    std::cout.precision(16);
    for (size_t i = 0; i < roots.size(); ++i) {
        double real = roots[i].real();
        double imag = roots[i].imag();
        std::cout << "Root " << (i + 1) << ": " << real << " + " << imag << "i\n";
        file << real << "," << imag << "\n";
    }

    file.close();
    std::cout << "Roots written to: " << output_filename << "\n";
}


Equation parse_equation(const std::string& input) {
    Equation result;
    std::string trimmed = input.substr(1, input.size() - 2);
    std::vector<std::string> term_strs = split_top_level(trimmed);
    for (const auto& tstr : term_strs) {
        parse_and_expand_term(tstr, result);
    }
    merge_terms(result);
    return result;
}


std::string substitute_expression(const std::string& input, const std::map<int, std::string>& hmap, int current_index) {
    std::regex tuple_regex(R"(\(([^;]+);\(([^,]+),([^)]+)\)\))");
    std::smatch match;

    std::string result = "{";
    std::string::const_iterator search_start(input.begin());
    bool first_tuple = true;

    while (std::regex_search(search_start, input.cend(), match, tuple_regex)) {
        std::string indices = match[1];
        std::string real = match[2];
        std::string imag = match[3];

        std::vector<std::string> parts = split_top_level(indices);
        std::string replaced_indices;
        bool first_index = true;

        for (const std::string& part : parts) {
		    int num = std::stoi(part);
		    if (!first_index) replaced_indices += ",";

		    if (num >= 2 && hmap.count(num - 2)) {
		        std::string expr = hmap.at(num - 2);
		        if (!expr.empty() && expr.front() == '{' && expr.back() == '}') {
		            replaced_indices += expr;
		        } else {
		            replaced_indices += "{" + expr + "}";
		        }
		    } else {
		        replaced_indices += part;
		    }
		    first_index = false;
		}

        if (!first_tuple) result += ",";
        result += "(" + replaced_indices + ";(" + real + "," + imag + "))";

        search_start = match.suffix().first;
        first_tuple = false;
    }

    result += "}";
    return result;
}


std::pair<std::string, int> process_equation_string(const std::string& input) {
    std::regex tuple_regex(R"(\(([^;]+);\(([-\d.]+),([-\d.]+)\)\))");
	std::smatch match;
    std::string modified = "{";
    std::vector<std::pair<std::string, std::complex<double>>> tuples;
    std::vector<std::vector<int>> index_vectors;

    auto it = input.begin();
    auto end = input.end();
    while (std::regex_search(it, end, match, tuple_regex)) {
		std::string indices_str = match[1].str();
		double real = std::stod(match[2].str());
		double imag = std::stod(match[3].str());
		std::complex<double> coeff(real, imag);

		std::vector<int> indices;
        std::stringstream ss(indices_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            indices.push_back(std::stoi(token));
        }

        index_vectors.push_back(indices);
        tuples.emplace_back(indices_str, coeff);
        it = match.suffix().first;
    }

    // Find max index among all values
    int max_index = -1;
	for (size_t i = 0; i < index_vectors.size(); ++i) {
	    const auto& idx_vec = index_vectors[i];
	    const auto& coeff = tuples[i].second;

	    // Skip 1-element (x;1) complex terms when computing max_index
	    if (idx_vec.size() == 1 && std::real(coeff) == 1.0 && std::imag(coeff) == 0.0)
	        continue;

	    for (int v : idx_vec) {
	        if (v > max_index) max_index = v;
	    }
	}

    // Remove single-index tuple (X;1) where X > all other indices
    int removed_index = -1;
    int remove_pos = -1;
    for (size_t i = 0; i < tuples.size(); ++i) {
        const auto& coeff = tuples[i].second;
        const auto& idx_vec = index_vectors[i];
        if (idx_vec.size() == 1 && std::real(coeff) == 1.0 && std::imag(coeff) == 0.0 && idx_vec[0] > max_index) {
            removed_index = idx_vec[0];
            remove_pos = i;
            break;
        }
    }

    if (remove_pos != -1) {
        tuples.erase(tuples.begin() + remove_pos);
        index_vectors.erase(index_vectors.begin() + remove_pos);
    }

    // Reconstruct with proper sign flipping
    for (size_t i = 0; i < tuples.size(); ++i) {
        const auto& indices = index_vectors[i];
        const auto& coeff = tuples[i].second;

        bool all_zero = std::all_of(indices.begin(), indices.end(), [](int x) { return x == 0; });
        bool skip_sign_flip = (std::real(coeff) == 1.0 && std::imag(coeff) == 0.0 && all_zero);

        std::complex<double> new_coeff = skip_sign_flip ? coeff : -coeff;

        //modified += "(" + tuples[i].first + ";" + std::to_string(std::real(new_coeff)) + ")";
        modified += "(" + tuples[i].first + ";(" +
            std::to_string(std::real(new_coeff)) + "," +
            std::to_string(std::imag(new_coeff)) + "))";
		if (i < tuples.size() - 1) modified += ",";
    }

    modified += "}";
    return {modified, removed_index};
}


int main() {
	
    int n_max;
    std::cout << "Enter maximum value of n (n >= 0): ";
    std::cin >> n_max;

    if (n_max < 0) {
        std::cout << "Invalid input. Please enter n >= 0.\n";
        return 1;
    }

	std::vector<Equation> equations = compute_i_equations(n_max);
	std::map<int, std::string> hmap;

	for (auto& eq : equations) {
		std::cout << "Equation: " << eq.to_string() << std::endl;
		auto [result, index] = process_equation_string(eq.to_string());
		std::cout << "Index: " << index << std::endl;
		index -= 2;
		hmap[index] = result;
	}
	std::cout << "n: " << n_max << std::endl;
    std::cout << "Size: " << hmap.size() << std::endl;

	for (int i = 0; i < n_max; ++i) {
	    std::string substituted = substitute_expression(hmap[i], hmap, i);
	    hmap[i] = substituted;
	}

	// Parse equation before solving it.
	for (int i = 0; i < n_max; i++) {
		std::string last = hmap[i]; // Solves G_{index + 2}
		Equation master = parse_equation(last);
		std::cout << "Master Equation: " << master.to_string() << std::endl;
		solve_polynomial(master, "roots.csv");
	}

	return 0;             
}

