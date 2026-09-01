"""
BPA Spare Parts Feasibility Model

This module implements the feasibility model for BluePrint Automation (BPA)
spare parts inventory management with centralized stocking.

Model Overview:
- Determine optimal base-stock levels S_i* at BPA (Poisson demand, service constraint)
- Compute per-component feasible interval [α_L,i, α_U,i] for subscription percentage α
- Check whether a universal α exists across all components

Sets & indices
--------------
i ∈ I          : spare parts
n ∈ N_i        : customers subscribed to component i  (those with λ_in > 0)
N_i = |N_i|    : number of subscriptions for component i

Parameters
----------
λ_in           : annual demand of customer n for component i
Λ_i^BPA        : = Σ_{n∈N_i} λ_in  (total BPA demand)
L_i            : lead time (years)
X              : target service level

U_i^BPA        : BPA purchase price
U_i^c          : customer sales price
α              : single subscription percentage  →  p_i = α · U_i^c

κ_BPA          : = c_f^BPA + c_h^BPA + c_o^BPA  (BPA carrying rate, incl. obsolescence)
κ_c            : = c_f^c  + c_h^c  + c_o^c      (customer carrying rate, incl. obsolescence)

BPA cost per component
----------------------
C_i^BPA(S) = κ_BPA · U_i^BPA · S
S_i* = min S ∈ Z+  s.t.  β_i(S) ≥ X

Customer self-stocking benchmark  (fixed S=1)
---------------------------------------------
C_in^self = κ_c · U_i^c

Revenue per component
---------------------
R_i = N_i · α · U_i^c

Feasible interval for α per component
--------------------------------------
α_L,i = C_i^BPA / (N_i · U_i^c)
α_U,i = min_{n∈N_i}(C_in^self) / U_i^c

Universal α:  max_i(α_L,i) ≤ α ≤ min_i(α_U,i)
"""

import numpy as np
from functools import lru_cache
from scipy.stats import poisson as _poisson


class BPAOptimizationModel:
    """
    Feasibility model for BPA spare parts service management.

    Attributes:
        sets       : customers, spare_parts
        parameters : all model parameters
    """

    def __init__(self):
        self.sets = {
            'customers':  [],
            'spare_parts': [],
            # kept for backward-compat with bpa_beheer / base_stock_overview
            'machines_per_customer': {},
        }
        self.parameters = {
            # --- decision ---
            'alpha':         None,   # α  subscription percentage
            'service_level': None,   # X  target fill rate
            # --- carrying rates ---
            'kappa_bpa': None,       # κ_BPA = c_f^BPA + c_h^BPA + c_o^BPA
            'kappa_c':   None,       # κ_c   = c_f^c  + c_h^c  + c_o^c
            # --- per-part obsolescence overrides (add on top of kappa) ---
            'obsolescence_rate_bpa': {},   # {part: extra obs rate for BPA}
            'obsolescence_rate_c':   {},   # {part: extra obs rate for customer}
            # --- prices ---
            'purchase_price': {},    # U_i^BPA
            'sales_price':    {},    # U_i^c
            # --- demand & lead time ---
            'lambda':    {},         # {(part, customer): λ_in}
            'lead_time': {},         # {part: L_i (years)}
            # --- backward-compat aliases ---
            'beta_target': None,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Setup helpers
    # ──────────────────────────────────────────────────────────────────────

    def add_customers_and_machines(self, customer_data):
        """
        Register customers.

        Args:
            customer_data (dict): {customer_id: number_of_machines}
                                  machine count is accepted but ignored by the
                                  new model (kept for backward-compatibility).
        """
        self.sets['customers'] = list(customer_data.keys())
        for customer, num_machines in customer_data.items():
            self.sets['machines_per_customer'][customer] = list(range(
                num_machines if isinstance(num_machines, int) else 1
            ))

    def add_spare_parts(self, part_list):
        self.sets['spare_parts'] = list(part_list)

    def add_parameters(self,
                       alpha=None,
                       service_level=None,
                       beta_target=None,
                       kappa_bpa=None,
                       kappa_c=None,
                       demand_data=None,
                       cost_data=None,
                       # legacy / convenience shorthands
                       holding_cost_percentage=None,
                       demand_multiplier=1.0,
                       price_percentage=None,
                       price=None):
        """
        Set model parameters.

        Primary parameters
        ------------------
        alpha            : subscription percentage α (e.g. 0.10)
        service_level    : target fill rate X (e.g. 0.99)
        kappa_bpa        : BPA carrying rate κ_BPA = c_f^BPA + c_h^BPA + c_o^BPA
        kappa_c          : customer carrying rate κ_c = c_f^c + c_h^c + c_o^c
        demand_data      : {(part, customer): λ_in}
        cost_data        : dict with any of:
            purchase_price          {part: U_i^BPA}
            sales_price             {part: U_i^c}
            lead_time               {part: L_i (years)}
            obsolescence_rate_bpa   {part: extra obs rate on top of kappa_bpa}
            obsolescence_rate_c     {part: extra obs rate on top of kappa_c}

        Legacy / convenience
        --------------------
        holding_cost_percentage : if kappa_bpa/kappa_c not given, both are set to
                                  this value (backward-compat with old call sites)
        price_percentage        : alias for alpha
        demand_multiplier       : all λ_in scaled by this factor (default 1.0)
        """
        # --- service level ---
        sl = service_level if service_level is not None else beta_target
        self.parameters['service_level'] = sl
        self.parameters['beta_target'] = sl

        # --- alpha (accept legacy price_percentage too) ---
        effective_alpha = alpha if alpha is not None else price_percentage
        self.parameters['alpha'] = effective_alpha
        # keep legacy key so old print helpers still work
        self.parameters['price_percentage'] = effective_alpha

        # --- carrying rates ---
        if kappa_bpa is not None:
            self.parameters['kappa_bpa'] = kappa_bpa
        elif holding_cost_percentage is not None and self.parameters['kappa_bpa'] is None:
            self.parameters['kappa_bpa'] = holding_cost_percentage

        if kappa_c is not None:
            self.parameters['kappa_c'] = kappa_c
        elif holding_cost_percentage is not None and self.parameters['kappa_c'] is None:
            self.parameters['kappa_c'] = holding_cost_percentage

        # --- demand (scale by multiplier) ---
        if demand_data:
            if demand_multiplier != 1.0:
                self.parameters['lambda'] = {k: v * demand_multiplier
                                             for k, v in demand_data.items()}
            else:
                self.parameters['lambda'] = demand_data

        # --- cost_data ---
        if cost_data:
            direct_keys = {'purchase_price', 'sales_price', 'lead_time',
                           'obsolescence_rate_bpa', 'obsolescence_rate_c'}
            for key, value in cost_data.items():
                if key in direct_keys:
                    self.parameters[key] = value

    # ──────────────────────────────────────────────────────────────────────
    #  Internal demand helpers
    # ──────────────────────────────────────────────────────────────────────

    def _subscribed_customers(self, part):
        """N_i: customers with λ_in > 0 for this part."""
        return [c for c in self.sets['customers']
                if self.parameters['lambda'].get((part, c), 0.0) > 0.0]

    def _total_demand(self, part):
        """Λ_i^BPA = Σ_{n∈N_i} λ_in."""
        return sum(self.parameters['lambda'].get((part, c), 0.0)
                   for c in self.sets['customers'])

    def _kappa_bpa_eff(self, part):
        """κ_BPA + per-part BPA obsolescence override."""
        return (self.parameters['kappa_bpa'] or 0.0) + \
               self.parameters['obsolescence_rate_bpa'].get(part, 0.0)

    def _kappa_c_eff(self, part):
        """κ_c + per-part customer obsolescence override."""
        return (self.parameters['kappa_c'] or 0.0) + \
               self.parameters['obsolescence_rate_c'].get(part, 0.0)

    # ──────────────────────────────────────────────────────────────────────
    #  Service-level functions (static, unchanged)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def service_level(s, lambda_rate, lead_time):
        """
        β_i(s; Λ, L) = P(Poisson(ΛL) < s) = Σ_{x=0}^{s-1} e^{-μ} μ^x / x!

        Args:
            s (int): base-stock level
            lambda_rate (float): demand rate Λ
            lead_time (float): lead time L (years)

        Returns:
            float: service level ∈ [0, 1]
        """
        if s <= 0:
            return 0.0
        mu = lambda_rate * lead_time
        try:
            return float(min(_poisson.cdf(s - 1, mu), 1.0))
        except Exception:
            return 0.95

    @staticmethod
    @lru_cache(maxsize=100_000)
    def inverse_service_level(target_service, lambda_rate, lead_time, max_s=300):
        """
        Minimum S such that β_i(S; Λ, L) ≥ target_service.
        Uses binary search.

        Gememoïseerd: in een overzicht-pass komen typisch duizenden componenten
        voor met slechts enkele tientallen unieke (λ, L)-combinaties. De
        lru_cache hergebruikt die resultaten, wat de Poisson-berekening (de
        dominante kost bij het laden) met een factor ~40 versnelt. De functie
        is puur (deterministisch, geen side-effects), dus caching is veilig.

        Args:
            target_service (float): target fill rate X
            lambda_rate (float): demand rate Λ
            lead_time (float): lead time L (years)
            max_s (int): search upper bound

        Returns:
            int: minimum base-stock level
        """
        if lambda_rate <= 0 or lead_time <= 0:
            return 0
        if BPAOptimizationModel.service_level(max_s, lambda_rate, lead_time) < target_service:
            return max_s
        left, right, result = 0, max_s, max_s
        while left <= right:
            mid = (left + right) // 2
            if BPAOptimizationModel.service_level(mid, lambda_rate, lead_time) >= target_service:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        return max(0, result)

    # ──────────────────────────────────────────────────────────────────────
    #  Core cost calculations
    # ──────────────────────────────────────────────────────────────────────

    def _optimal_base_stock(self, part):
        """S_i* = min S ∈ Z+  s.t.  β_i(S; Λ_i^BPA, L_i) ≥ X."""
        lambda_bpa = self._total_demand(part)
        lead_time  = self.parameters['lead_time'].get(part, 0.0)
        sl         = self.parameters['service_level'] or 0.95
        return self.inverse_service_level(sl, lambda_bpa, lead_time)

    def _bpa_cost_per_part(self, part):
        """C_i^BPA = κ_BPA_eff · U_i^BPA · S_i*"""
        lambda_bpa = self._total_demand(part)
        if lambda_bpa == 0.0:
            return 0.0, 0
        kappa  = self._kappa_bpa_eff(part)
        u_bpa  = self.parameters['purchase_price'].get(part, 0.0)
        s_star = self._optimal_base_stock(part)
        return kappa * u_bpa * s_star, s_star

    def _customer_self_stocking_cost(self, part, customer):
        """C_in^self = κ_c_eff · U_i^c  (fixed S=1, no penalty)"""
        lambda_in = self.parameters['lambda'].get((part, customer), 0.0)
        if lambda_in == 0.0:
            return 0.0
        kappa_c = self._kappa_c_eff(part)
        u_c     = self.parameters['sales_price'].get(part, 0.0)
        return kappa_c * u_c

    # ──────────────────────────────────────────────────────────────────────
    #  Alpha interval analysis  (core output)
    # ──────────────────────────────────────────────────────────────────────

    def calculate_alpha_intervals(self):
        """
        Compute per-component feasible interval [α_L,i, α_U,i] and universal α.

        For each component i:
            α_L,i = C_i^BPA / (N_i · U_i^c)          BPA cost recovery
            α_U,i = min_{n∈N_i}(C_in^self) / U_i^c   customer attractiveness

        Universal:
            α_L = max_i(α_L,i),  α_U = min_i(α_U,i)

        Returns:
            dict with:
                per_component   : {part: {N_i, C_i_BPA, S_star, min_self_cost,
                                          alpha_L, alpha_U, feasible}}
                universal_alpha_L
                universal_alpha_U
                universal_feasible
        """
        per = {}
        for part in self.sets['spare_parts']:
            subscribed = self._subscribed_customers(part)
            N_i = len(subscribed)
            u_c = self.parameters['sales_price'].get(part, 0.0)

            c_bpa, s_star = self._bpa_cost_per_part(part)

            if N_i == 0 or u_c == 0.0:
                per[part] = {
                    'N_i': N_i, 'C_i_BPA': c_bpa, 'S_star': s_star,
                    'min_self_cost': None, 'alpha_L': None, 'alpha_U': None,
                    'feasible': False,
                }
                continue

            self_costs = [self._customer_self_stocking_cost(part, n) for n in subscribed]
            min_self   = min(self_costs)

            alpha_L = c_bpa / (N_i * u_c)
            alpha_U = min_self / u_c

            per[part] = {
                'N_i':           N_i,
                'C_i_BPA':       c_bpa,
                'S_star':        s_star,
                'min_self_cost': min_self,
                'alpha_L':       alpha_L,
                'alpha_U':       alpha_U,
                'feasible':      alpha_L <= alpha_U,
            }

        # Universal α
        valid = [v for v in per.values() if v['alpha_L'] is not None]
        if valid:
            alpha_L_univ = max(v['alpha_L'] for v in valid)
            alpha_U_univ = min(v['alpha_U'] for v in valid)
        else:
            alpha_L_univ = alpha_U_univ = None

        return {
            'per_component':     per,
            'universal_alpha_L': alpha_L_univ,
            'universal_alpha_U': alpha_U_univ,
            'universal_feasible': (
                alpha_L_univ is not None and alpha_L_univ <= alpha_U_univ
            ),
        }

    # ──────────────────────────────────────────────────────────────────────
    #  check_feasibility  (uses α from parameters)
    # ──────────────────────────────────────────────────────────────────────

    def check_feasibility(self):
        """
        Check model feasibility at the configured α.

        BPA profitable     : Σ_i R_i  > Σ_i C_i^BPA       with R_i = N_i · α · U_i^c
        Customer benefit   : α · U_i^c ≤ C_in^self  ∀ i,n

        Returns:
            dict with keys: feasible, bpa_profitable, bpa_margin, all_customers_benefit,
                            total_revenue, bpa_costs, revenue_by_part,
                            customer_benefits, alpha, service_level,
                            price_percentage (alias for alpha),
                            alpha_intervals
        """
        alpha = self.parameters.get('alpha') or 0.0

        bpa_costs_total  = 0.0
        total_revenue    = 0.0
        revenue_by_part  = {}

        for part in self.sets['spare_parts']:
            c_bpa, _ = self._bpa_cost_per_part(part)
            bpa_costs_total += c_bpa

            subscribed = self._subscribed_customers(part)
            u_c = self.parameters['sales_price'].get(part, 0.0)
            rev = len(subscribed) * alpha * u_c
            revenue_by_part[part] = rev
            total_revenue += rev

        bpa_margin    = total_revenue - bpa_costs_total
        bpa_profitable = bpa_margin > 0

        customer_benefits = {}
        for customer in self.sets['customers']:
            self_cost_total    = 0.0
            subscription_total = 0.0
            for part in self.sets['spare_parts']:
                lambda_in = self.parameters['lambda'].get((part, customer), 0.0)
                if lambda_in == 0.0:
                    continue
                u_c = self.parameters['sales_price'].get(part, 0.0)
                self_cost_total    += self._customer_self_stocking_cost(part, customer)
                subscription_total += alpha * u_c

            savings = self_cost_total - subscription_total
            customer_benefits[customer] = {
                'self_stocking_cost': self_cost_total,
                'bpa_service_cost':   subscription_total,
                'savings':            savings,
                'benefits':           savings > 0,
            }

        all_customers_benefit = all(v['benefits'] for v in customer_benefits.values())

        return {
            'feasible':              bpa_profitable and all_customers_benefit,
            'bpa_profitable':        bpa_profitable,
            'bpa_margin':            bpa_margin,
            'all_customers_benefit': all_customers_benefit,
            'total_revenue':         total_revenue,
            'bpa_costs':             bpa_costs_total,
            'revenue_by_part':       revenue_by_part,
            'customer_benefits':     customer_benefits,
            'alpha':                 alpha,
            'price_percentage':      alpha,   # backward-compat alias
            'service_level':         self.parameters.get('service_level'),
            'alpha_intervals':       self.calculate_alpha_intervals(),
        }

    # ──────────────────────────────────────────────────────────────────────
    #  calculate_base_stock_levels  (backward-compat for bpa_beheer etc.)
    # ──────────────────────────────────────────────────────────────────────

    def calculate_base_stock_levels(self):
        """
        Return {part: S_i*} for all spare parts.
        """
        return {part: self._optimal_base_stock(part)
                for part in self.sets['spare_parts']}

    def print_base_stock_levels(self):
        bsl = self.calculate_base_stock_levels()
        print("\n" + "=" * 70)
        print("BASE STOCK LEVELS AT BPA LOCATION")
        print("=" * 70)
        print(f"{'Part ID':<22} {'Λ_i^BPA':>12} {'L_i (yr)':>10} {'S_i*':>8}")
        print("-" * 70)
        for part in self.sets['spare_parts']:
            lam = self._total_demand(part)
            lt  = self.parameters['lead_time'].get(part, 0.0)
            print(f"{part:<22} {lam:>12.4f} {lt:>10.4f} {bsl[part]:>8}")
        print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    #  calculate_detailed_bpa_costs  (backward-compat)
    # ──────────────────────────────────────────────────────────────────────

    def calculate_detailed_bpa_costs(self):
        """
        Returns {part: {'demand', 'base_stock', 'inventory_cost', 'total'}}.
        """
        result = {}
        for part in self.sets['spare_parts']:
            lam = self._total_demand(part)
            if lam == 0.0:
                result[part] = {'demand': 0, 'base_stock': 0,
                                'inventory_cost': 0, 'total': 0}
                continue
            cost, s_star = self._bpa_cost_per_part(part)
            result[part] = {
                'demand':         lam,
                'base_stock':     s_star,
                'inventory_cost': cost,
                'total':          cost,
            }
        return result

    # ──────────────────────────────────────────────────────────────────────
    #  calculate_detailed_customer_costs  (backward-compat)
    # ──────────────────────────────────────────────────────────────────────

    def calculate_detailed_customer_costs(self):
        """
        Returns {customer: {part: C_in^self, 'total': X}}.
        """
        result = {}
        for customer in self.sets['customers']:
            result[customer] = {}
            total = 0.0
            for part in self.sets['spare_parts']:
                c = self._customer_self_stocking_cost(part, customer)
                result[customer][part] = c
                total += c
            result[customer]['total'] = total
        return result

    # ──────────────────────────────────────────────────────────────────────
    #  print_detailed_costs
    # ──────────────────────────────────────────────────────────────────────

    def print_detailed_costs(self):
        bpa_det   = self.calculate_detailed_bpa_costs()
        cust_det  = self.calculate_detailed_customer_costs()
        intervals = self.calculate_alpha_intervals()

        # ── BPA costs ─────────────────────────────────────────────────────
        print("\n" + "=" * 92)
        print("BPA CENTRALE VOORRAADKOSTEN PER ONDERDEEL")
        print("=" * 92)
        print(f"{'Part':<22} {'Λ_BPA':>10} {'μ=Λ·L':>8} {'S*':>5} {'κ_BPA_eff':>11} {'C_BPA':>12} "
              f"{'α_L,i':>8} {'α_U,i':>8} {'OK':>4}")
        print("-" * 92)
        total_bpa = 0.0
        for part in self.sets['spare_parts']:
            d   = bpa_det[part]
            iv  = intervals['per_component'].get(part, {})
            al  = f"{iv.get('alpha_L', 0):.2%}" if iv.get('alpha_L') is not None else '—'
            au  = f"{iv.get('alpha_U', 0):.2%}" if iv.get('alpha_U') is not None else '—'
            ok  = '✓' if iv.get('feasible') else '✗'
            kap = self._kappa_bpa_eff(part)
            lt  = self.parameters['lead_time'].get(part, 0.0)
            mu  = d['demand'] * lt
            print(f"{part:<22} {d['demand']:>10.4f} {mu:>8.4f} {d['base_stock']:>5} "
                  f"{kap:>10.2%} €{d['total']:>10.2f} {al:>8} {au:>8} {ok:>4}")
            total_bpa += d['total']
        print("-" * 92)
        print(f"{'TOTAAL BPA KOSTEN':<58} €{total_bpa:.2f}")

        # ── Universal α ───────────────────────────────────────────────────
        al_u = intervals['universal_alpha_L']
        au_u = intervals['universal_alpha_U']
        print(f"\n  Universeel interval α:  "
              f"[{al_u:.4%}, {au_u:.4%}]  "
              f"({'HAALBAAR' if intervals['universal_feasible'] else 'NIET HAALBAAR'})")
        print("=" * 80 + "\n")

        # ── Customer self-stocking costs ───────────────────────────────────
        print("=" * 100)
        print("CUSTOMER SELF-STOCKING COSTS (C_in^self = κ_c_eff · U_i^c)")
        print("=" * 100)
        hdr = f"{'Klant':<15}"
        for part in self.sets['spare_parts']:
            hdr += str(part)[:14].ljust(16)
        hdr += "Totaal"
        print(hdr)
        print("-" * 100)
        total_cust = 0.0
        for customer in self.sets['customers']:
            row = f"{customer:<15}"
            for part in self.sets['spare_parts']:
                row += f"€{cust_det[customer].get(part, 0):<14.2f}"
            ct = cust_det[customer]['total']
            row += f"€{ct:.2f}"
            print(row)
            total_cust += ct
        print("-" * 100)
        print(f"{'TOTAAL':<15}" + "".ljust(16 * len(self.sets['spare_parts'])) + f"€{total_cust:.2f}")
        print("=" * 100 + "\n")

        # ── Summary ───────────────────────────────────────────────────────
        print("=" * 70)
        print("COST SUMMARY")
        print("=" * 70)
        print(f"  Total BPA Stocking Costs    : €{total_bpa:>12.2f}")
        print(f"  Total Customer Self-Stocking: €{total_cust:>12.2f}")
        print(f"  Combined                    : €{total_bpa + total_cust:>12.2f}")
        print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────
    #  Backward-compat: calculate_revenue_per_part
    # ──────────────────────────────────────────────────────────────────────

    def calculate_revenue_per_part(self):
        """R_i = N_i · α · U_i^c  for each part."""
        alpha = self.parameters.get('alpha') or 0.0
        revenue_by_part = {}
        total = 0.0
        for part in self.sets['spare_parts']:
            n_i = len(self._subscribed_customers(part))
            u_c = self.parameters['sales_price'].get(part, 0.0)
            rev = n_i * alpha * u_c
            revenue_by_part[part] = rev
            total += rev
        return revenue_by_part, total


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BPAOptimizationModel()
    print("BPA Optimization Model initialized")