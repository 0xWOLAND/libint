#include "catch.hpp"
#include "fixture.h"
#include <libint2.hpp>
#include <libint2/solidharmonics.h>
using namespace libint2;

#include <cmath>
#include <iomanip>
#include <iostream>

// ---- CI diagnostic (temporary): solid-harmonics coefficients and the
// ---- opop_coulomb_opop bra-ket swap path, printed without references
TEST_CASE("DIAG windows", "[diag]") {
  using libint2::solidharmonics::SolidHarmonicsCoefficients;
  std::cout << std::setprecision(10);
  std::cout << "DIAG parity(-1)=" << int(parity(-1)) << " parity(1)=" << int(parity(1))
            << " parity(-2/2)=" << int(parity(-2 / 2)) << " sho=" << int(libint2::solid_harmonics_ordering())
            << " cartorder=" << LIBINT_CGSHELL_ORDERING << "\n";
  auto dump = [](const char* what, const SolidHarmonicsCoefficients<double>& c, int l) {
    std::cout << "DIAG " << what << " l=" << l << ":";
    for (int s = 0; s < 2 * l + 1; ++s) {
      std::cout << " row" << s << "{";
      for (unsigned k = 0; k < c.nnz(s); ++k) std::cout << " " << c.row_idx(s)[k] << ":" << c.row_values(s)[k];
      std::cout << " }";
    }
    std::cout << "\n";
  };
  dump("instance", SolidHarmonicsCoefficients<double>::instance(2), 2);
  dump("make_instance", SolidHarmonicsCoefficients<double>::make_instance(2), 2);
  dump("instance", SolidHarmonicsCoefficients<double>::instance(1), 1);
  std::cout << "DIAG coeff(2,0,xx)=" << SolidHarmonicsCoefficients<double>::coeff(2, 0, 2, 0, 0)
            << " coeff(2,0,yy)=" << SolidHarmonicsCoefficients<double>::coeff(2, 0, 0, 2, 0)
            << " coeff(2,0,zz)=" << SolidHarmonicsCoefficients<double>::coeff(2, 0, 0, 0, 2) << "\n";
#if defined(LIBINT2_SUPPORT_ONEBODY)
  {
    std::vector<Shell> c{Shell{{1.0}, {{1, false, {10.0}}}, {{0.0, 0.0, 0.0}}},
                         Shell{{1.0}, {{2, false, {10.0}}}, {{0.1, 0.2, 0.3}}}};
    std::vector<Shell> p{Shell{{1.0}, {{1, true, {10.0}}}, {{0.0, 0.0, 0.0}}},
                         Shell{{1.0}, {{2, true, {10.0}}}, {{0.1, 0.2, 0.3}}}};
    Engine eng(Operator::overlap, 2, 2);
    eng.compute(c[0], c[1]);
    std::cout << "DIAG cart<p|d>:"; for (int i = 0; i < 18; ++i) std::cout << " " << eng.results()[0][i]; std::cout << "\n";
    eng.compute(p[0], p[1]);
    std::cout << "DIAG pure<p|d>:"; for (int i = 0; i < 15; ++i) std::cout << " " << eng.results()[0][i]; std::cout << "\n";
  }
#endif
#if LIBINT2_TASK_EXISTS_4opop_coulomb_opop
  {
    // (s d|s p) is canonicalized by the engine via braket swap + coupled pair
    // swap onto the kernel (p s|d s), which needs no permutation at all.
    // Check the two against each other: A[c][i*3+j] == phase(c) * B[T(c)][j*6+i]
    Shell s{{1.0}, {{0, false, {1.0}}}, {{0.0, 0.0, 0.0}}};
    Shell d{{1.2}, {{2, false, {1.0}}}, {{0.3, -0.8, 0.4}}};
    Shell pp{{2.0}, {{1, false, {1.0}}}, {{1.0, 1.0, 1.0}}};
    Engine eng(Operator::opop_coulomb_opop, 1, 2, 0);
    const auto& A0 = eng.compute(s, d, s, pp);
    std::vector<std::vector<double>> A; for (auto* t : A0) A.emplace_back(t, t + 18);
    const auto& B0 = eng.compute(pp, s, d, s);
    std::vector<std::vector<double>> B; for (auto* t : B0) B.emplace_back(t, t + 18);
    double maxdev = 0; int worst_c = -1, worst_e = -1;
    for (int cc = 0; cc < 16; ++cc) {
      const int T = 4 * (cc % 4) + cc / 4;
      const double phase = ((cc / 4 > 0) != (cc % 4 > 0)) ? -1.0 : 1.0;
      for (int i = 0; i < 6; ++i) for (int j = 0; j < 3; ++j) {
        const double dev = std::abs(A[cc][i * 3 + j] - phase * B[T][j * 6 + i]);
        if (dev > maxdev) { maxdev = dev; worst_c = cc; worst_e = i * 3 + j; }
      }
    }
    std::cout << "DIAG opop (s d|s p) vs (p s|d s): maxdev=" << maxdev << " at comp=" << worst_c << " elem=" << worst_e << "\n";
    std::cout << "DIAG opop A[1][0..5]:"; for (int e = 0; e < 6; ++e) std::cout << " " << A[1][e]; std::cout << "\n";
    std::cout << "DIAG opop B[4][*]:"; for (int e = 0; e < 18; ++e) std::cout << " " << B[4][e]; std::cout << "\n";
  }
#endif
}
