#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <regex>
#include <numeric>
#include <cmath>
#include <complex>

struct Term {
    std::vector<int> indices;
    long long coefficient;

    Term(std::vector<int> idx, long long coefficient) : indices(idx), coefficient(coefficient) {}

    friend std::ostream& operator<<(std::ostream& os, const Term& t) {
        os << "(";
        
        size_t i = 0;
	    while (i < t.indices.size() - 1) {
	        os << t.indices[i] << ",";
	        i++;
	    }
		if (i == t.indices.size() - 1) { 
			os << t.indices[i] << ";" << t.coefficient << ")";
		}

        return os;
    }

	std::string to_string() const {
        std::ostringstream oss;
        oss << "(";

        size_t i = 0;
	    while (i < this->indices.size() - 1) {
	        oss << this->indices[i] << ",";
	        i++;
	    }
		if (i == this->indices.size() - 1) { 
			oss << this->indices[i] << ";" << this->coefficient << ")";
		}

        return oss.str();
	}
};


struct Equation {
    std::vector<Term> terms;

    friend std::ostream& operator<<(std::ostream& os, const Equation& eq) {
        os << "{";
    	
		size_t i = 0;
        while (i < eq.terms.size() - 1) {
			os << eq.terms[i] << ",";
            i++;
		}
		if (i == eq.terms.size() - 1) { 
			os << eq.terms[i] << "}";
	    }		

        return os;
    }

	std::string to_string() const {
		std::ostringstream oss;
		oss << "{";

		size_t i = 0;
        while (i < this->terms.size() - 1) {
			oss << this->terms[i] << ",";
            i++;
		}
		if (i == this->terms.size() - 1) { 
			oss << this->terms[i] << "}";
	    }		

        return oss.str();
	}
};
            


std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \n\r\t");
    size_t end = s.find_last_not_of(" \n\r\t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}                                               


std::string substitute(std::string equation, std::map<int, std::string>& equations) {

    std::regex tuple_regex(R"(\(\d+(?:,\d+)*;\s*[-]?\d+\))");
    std::regex number_regex(R"(\d+)");
    std::smatch match;

    std::string m_equation = "{";
	std::string::const_iterator search_start(equation.cbegin());
   
 	size_t total_matches = std::distance(std::sregex_iterator(equation.begin(), equation.end(), tuple_regex), std::sregex_iterator());
	size_t current_match = 0;

	bool last_tuple = false;
    while (std::regex_search(search_start, equation.cend(), match, tuple_regex)) {
        std::string tuple = match[0];
        std::smatch num_match;

        size_t semicolon_pos = tuple.find(";");
        std::string before_semicolon = tuple.substr(1, semicolon_pos - 1);
        std::string after_semicolon = tuple.substr(semicolon_pos);

        m_equation += "(";
		std::string::const_iterator num_search_start(before_semicolon.cbegin());

        bool first_number = true;
        
		while (std::regex_search(num_search_start, before_semicolon.cend(), num_match, number_regex)) {
            int num = stoi(num_match[0]);
            std::string replacement = (num > 2) ? equations[num] : num_match.str();

            if (!first_number) m_equation += ",";
			first_number = false;

			m_equation += replacement;
            num_search_start = num_match.suffix().first;
        }
        m_equation += after_semicolon;
   
		bool last_tuple = (current_match == total_matches - 1);
        if (!last_tuple) m_equation += ",";

        search_start = match.suffix().first;
        current_match++;
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
    std::map<std::vector<int>, long long> merged_terms;  // Map to sum coefficients of identical terms

    for (const auto &term : eq.terms) {
        /*DEBUG if (std::abs(term.coefficient) > 1e9) {
            std::cout << "Large term: " << term.to_string() << std::endl;
        }*/
        std::vector<int> sorted_indices = term.indices;
        sort(sorted_indices.begin(), sorted_indices.end());  // Ensure consistent order

        merged_terms[sorted_indices] += term.coefficient;
    }

    eq.terms.clear();
    for (const auto &entry : merged_terms) {
        eq.terms.emplace_back(entry.first, static_cast<long long>(entry.second));
    }
}


std::vector<Equation> compute_equations(int m_max) {
    std::vector<Equation> equations;

	for (int m = 0; m <= m_max; m++) {
        Equation eq;
        
		eq.terms.emplace_back(std::vector<int>{2 * m + 4}, 1);
        for (int k = 1; k <= 2 * m + 1; k += 2) {
            int coeff = 3 * binomial(2 * m + 1, k);
			eq.terms.emplace_back(std::vector<int>{k + 1, 2 * m - k + 3}, coeff);
        }

		
        for (int k = 1; k <= 2 * m + 1; k += 2) {
            for (int l = 1; l <= 2 * m - k; l += 2) {
                int coeff = binomial(2 * m + 1, k) * binomial(2 * m + 1 - k, l);
				eq.terms.emplace_back(std::vector<int>{k + 1, l + 1, 2 * m + 2 - k - l}, coeff);
            }
        }

		merge_terms(eq);

		if (m == 0) {
			eq.terms.emplace_back(std::vector<int>{0,0}, 1);
		}
        equations.push_back(eq);
    }

    return equations;
}


Equation parse_equation(const std::string& input);


Equation expand_nested(const std::vector<int>& outer_indices, int scalar, const Equation& nested) {
    Equation result;
    for (const auto& term : nested.terms) {
        std::vector<int> indices = outer_indices;
        indices.insert(indices.end(), term.indices.begin(), term.indices.end());
        result.terms.emplace_back(indices, 1LL * scalar * term.coefficient);
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
    if (s.front() != '(' || s.back() != ')') throw std::runtime_error("Malformed term: " + s);
    std::string inner = s.substr(1, s.size() - 2);
    size_t semi = inner.rfind(';');
    std::string left = trim(inner.substr(0, semi));
    long long coeff = std::stoi(trim(inner.substr(semi + 1)));
	
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
                        next.terms.emplace_back(merged, 1LL * t1.coefficient * t2.coefficient);
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


std::string format_as_ratios(const Equation& eq) {
    if (eq.terms.empty()) return "{}";

    // Step 1: Find the coefficient of the term with highest power of x
    int max_xpow = -1;
    long long denom = 1;
    for (const auto& term : eq.terms) {
        int xpow = std::count(term.indices.begin(), term.indices.end(), 2);
        if (xpow > max_xpow) {
            max_xpow = xpow;
            denom = term.coefficient;
        }
    }

    std::ostringstream oss;
    oss << "{";

    for (size_t i = 0; i < eq.terms.size(); ++i) {
        const auto& term = eq.terms[i];
        oss << "(";

        // Write index list
        for (size_t j = 0; j < term.indices.size(); ++j) {
            oss << term.indices[j];
            if (j < term.indices.size() - 1) oss << ",";
        }

        oss << ";";

        // Step 2: Reduce numerator and denominator
        long long num = term.coefficient;
        long long g = std::gcd(num, denom);
        num /= g;
        long long reduced_denom = denom / g;

        // Step 3: Apply minus sign only on numerator
        if (reduced_denom == 1) {
            oss << num;
        } else {
            oss << num << "/" << reduced_denom;
        }

        oss << ")";
        if (i < eq.terms.size() - 1) oss << ",";
    }

    oss << "}";
    return oss.str();
}



void print_expression(const Equation& parsed) {
    std::cout << "Expression in X:" << std::endl;

    // Step 1: Find the highest power of X and its coefficient
    int max_pow = -1;
    long long denom = 1;
    for (const auto& term : parsed.terms) {
        int x_pow = std::count(term.indices.begin(), term.indices.end(), 2);
        if (x_pow > max_pow) {
            max_pow = x_pow;
            denom = term.coefficient;
        }
    }

    // Step 2: Format the expression
    for (size_t idx = 0; idx < parsed.terms.size(); ++idx) {
        const auto& term = parsed.terms[idx];

        int x_pow = std::count(term.indices.begin(), term.indices.end(), 2);

        long long g = std::gcd(term.coefficient, denom);
        long long num = term.coefficient / g;
        long long den = denom / g;

        // Handle sign and formatting
        if (idx > 0) {
            std::cout << (num >= 0 ? " + " : " - ");
        } else if (num < 0) {
            std::cout << "-";
        }

        num = std::abs(num);

        if (den == 1) {
            std::cout << num;
        } else {
            std::cout << num << "/" << den;
        }

        if (x_pow > 0) {
            std::cout << "*X";
            if (x_pow > 1) std::cout << "^" << x_pow;
        }
    }

    std::cout << " = 0" << std::endl;
}


void solve_polynomial(const Equation& parsed) {
    std::cout << "Roots of the polynomial:" << std::endl;

    // Determine max power of x
    int max_xpow = -1;
    long long denom = 1;
    for (const auto& term : parsed.terms) {
        int pow = std::count(term.indices.begin(), term.indices.end(), 2);
        if (pow > max_xpow) {
            max_xpow = pow;
            denom = term.coefficient;
        }
    }

    const int degree = max_xpow;
    std::vector<std::complex<double>> coeffs(degree + 1, 0.0);

    // Fill in normalized coefficients
    for (const auto& term : parsed.terms) {
        int x_pow = std::count(term.indices.begin(), term.indices.end(), 2);
        long long g = std::gcd(term.coefficient, denom);
        long long num = term.coefficient / g;
        long long den = denom / g;
        double value = static_cast<double>(num) / static_cast<double>(den);
        coeffs[x_pow] += std::complex<double>(value, 0.0);
    }

    // Remove trailing zeros
    int deg = degree;
    while (deg > 0 && std::abs(coeffs[deg]) < 1e-14) deg--;

    std::vector<std::complex<double>> roots(deg);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < deg; ++i) {
        double angle = 2 * pi * i / deg;
        roots[i] = std::polar(1.0, angle);
    }

    const int max_iter = 1000;
    const double tol = 1e-12;

    // Durand–Kerner method
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

    std::cout.precision(16);
    for (size_t i = 0; i < roots.size(); ++i) {
        std::cout << "Root " << (i + 1) << ": " << roots[i].real() << " + " << roots[i].imag() << "i" << std::endl;
    }
}


std::pair<std::string, int> process_equation_string(const std::string& input) {
    std::regex tuple_regex(R"(\(([^;]+);(-?\d+)\))");
    std::smatch match;
    std::string modified = "{";
    std::vector<std::pair<std::string, long long>> tuples;
    std::vector<std::vector<int>> index_vectors;

    auto it = input.begin();
    auto end = input.end();
    while (std::regex_search(it, end, match, tuple_regex)) {
        std::string indices_str = match[1].str();
        long long coeff = std::stoll(match[2].str());

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
    for (const auto& vec : index_vectors) {
        for (int v : vec) {
            if (v > max_index) max_index = v;
        }
    }

    // Remove single-index tuple (X;1) where X > any other index
    int removed_index = -1;
    int remove_pos = -1;
    for (size_t i = 0; i < tuples.size(); ++i) {
        const auto& [_, coeff] = tuples[i];
        const auto& idx_vec = index_vectors[i];
        if (idx_vec.size() == 1 && coeff == 1 && idx_vec[0] > max_index - 1) {
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
        long long coeff = tuples[i].second;

        bool all_zero = std::all_of(indices.begin(), indices.end(), [](int x) { return x == 0; });
        bool skip_sign_flip = (coeff == 1 && all_zero);

        long long new_coeff = skip_sign_flip ? coeff : -coeff;

        modified += "(" + tuples[i].first + ";" + std::to_string(new_coeff) + ")";
        if (i < tuples.size() - 1) modified += ",";
    }

    modified += "}";
    return {modified, removed_index};
}


// Stirling-type asymptotic approximation for Gamma(1/4)
constexpr double gamma_one_fourth_approx = 3.625609908221908311930685155867672;


/// Compute the alternative asymptotic approximation for G_{n_max / 2}
double compute_half_asymptotic(int n_max) {
    double n = static_cast<double>(n_max) / 2.0;
    double x = n/2 + 0.25;
    constexpr double pi = 3.14159265358979323846;
    double sqrt_pi = std::sqrt(pi);
    double power = std::pow(x, n / 2.0 - 0.25);
    double expo = std::exp(-x);
    double coeff = std::pow(2, n);
    double denom = 2.0 * std::pow(4.0, 0.75) * gamma_one_fourth_approx;

    return (coeff * sqrt_pi * power * expo); /// denom;
}


void prepend_asymptotic_term(Equation& eq, int n_max) {
    const double r = 0.4095057;
    int n = n_max/2;
    
    // Method BENDER.
    double sign = (n % 2 == 0) ? 1.0 : -1.0;

    // log factorial to avoid overflow
    double log_fact = 0.0;
    for (int i = 1; i <= 2 * n - 1; ++i) {
        log_fact += std::log(i);
    }

    double log_result = std::log(2.0) + (2 * n) * std::log(r) + log_fact;
    double result = std::exp(log_result) * ((sign < 0) ? -1.0 : 1.0);

    // Convert to long long with rounding if needed, or keep as double if mixed
    long long asymp_value = static_cast<long long>(result);
    std::cout << "Asymptotic value (BENDER): " << asymp_value << std::endl;

    // Method UGent.
    result = compute_half_asymptotic(n_max);
    asymp_value = static_cast<long long>(result);

    std::cout << "Asymptotic value (UGENT): " << asymp_value << std::endl;

    std::vector<int> zero_indices(n, 0);
    Term asymp_term(zero_indices, asymp_value);

    eq.terms.insert(eq.terms.begin(), asymp_term);
}


int main() {
    int n_max;
    std::cout << "Enter maximum value of n (n >= 0): ";
    std::cin >> n_max;

    if (n_max < 0) {
        std::cout << "Invalid input. Please enter n >= 0.\n";
        return 1;
    }

    std::vector<Equation> equations = compute_equations(n_max);
	std::map<int, std::string> hmap;
	
	// Init with G_0 and G_2.
	Equation G0, G2;
	G0.terms.push_back({{0}, 1});
	G2.terms.push_back({{2}, 1});
	hmap[0] = G0.to_string();
	hmap[2] = G2.to_string();
	
	for (auto& eq : equations) {
		std::cout << "Equation :" << eq.to_string() << std::endl;
		auto [result, index] = process_equation_string(eq.to_string());
    	//std::cout << "Processed: " << result << "\n";
    	//std::cout << "Removed index: " << index << "\n";
		hmap[index] = result;
	}
	
	std::cout << "CALCULATED" << std::endl;
    std::cout << hmap[0] << std::endl;
    std::cout << hmap[2] << std::endl;
    std::cout << hmap[4] << std::endl;
    std::cout << hmap[6] << std::endl;
	std::cout << hmap[8] << std::endl;
	std::cout << hmap[10] << std::endl;
	std::cout << hmap[12] << std::endl;
	std::cout << hmap[14] << std::endl;
    std::cout << hmap[16] << std::endl;

    std::map<int, std::string> m;
	for (int i = 0; i <= n_max; i += 2) {
    	std::string test = substitute(hmap[i], hmap);
		if (test != hmap[i]) hmap[i] = test;
	}

	std::cout << "ALTERED" << std::endl; 
    std::cout << hmap[0] << std::endl;
    std::cout << hmap[2] << std::endl;
    std::cout << hmap[4] << std::endl;
	std::cout << hmap[6] << std::endl;
	std::cout << hmap[8] << std::endl;
	std::cout << hmap[10] << std::endl;
	std::cout << hmap[12] << std::endl;
    std::cout << hmap[14] << std::endl;
    std::cout << hmap[16] << std::endl;	

	std::string after = hmap[n_max];
	std::cout << "Before parse_equation" << std::endl;


    int method;
    std::cout << "Select method to solve Dyson-Schwinger equation:\n";
    std::cout << "1. Hard truncation\n2. Asymptotics\nEnter choice (1 or 2): ";
    std::cin >> method;
    std::cin.ignore();

    if (method == 1) {
        Equation parsed = parse_equation(after);
        std::cout << "Parsed and expanded: " << parsed.to_string() << std::endl;
        std::cout << "Normalized: " << format_as_ratios(parsed) << std::endl;

        print_expression(parsed);
        std::cout << "Equation: " << parsed.to_string() << std::endl;
        solve_polynomial(parsed); 
    } else if (method == 2) {
        Equation parsed = parse_equation(after);
        prepend_asymptotic_term(parsed, n_max);
        std::cout << "Equation with asymptotic constant added: " << parsed.to_string() << std::endl;
        std::cout << "Normalized: " << format_as_ratios(parsed) << std::endl;
        print_expression(parsed);
        solve_polynomial(parsed);
    } else {
        std::cerr << "Invalid choice." << std::endl;
        return 1;
    }

    std::cout << "Output for: " << n_max << std::endl;

	return 0;             
}