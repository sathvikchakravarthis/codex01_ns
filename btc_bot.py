from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging


@dataclass
class RiskManagementConfig:
    risk_per_trade_percent: float
    max_daily_drawdown_percent: float
    max_total_drawdown_percent: float
    cooldown_after_loss_minutes: int
    min_rr_ratio: float


@dataclass
class PositionSizingConfig:
    method: str
    atr_multiplier: float
    fee_buffer_percent: float
    slippage_buffer_percent: float


@dataclass
class EMAConfig:
    fast: int
    slow: int
    trend_filter: int


@dataclass
class RSIConfig:
    period: int
    overbought: float
    oversold: float


@dataclass
class MACDConfig:
    fast: int
    slow: int
    signal: int


@dataclass
class ATRConfig:
    period: int


@dataclass
class VolumeConfig:
    ma_period: int
    spike_multiplier: float


@dataclass
class EntryLongConfig:
    trend_confirmation: bool
    momentum_confirmation: bool
    volume_confirmation: bool
    rsi_min: float
    rsi_max: float


@dataclass
class StopLossConfig:
    type: str
    atr_multiplier: float


@dataclass
class TakeProfitConfig:
    type: str
    rr_ratio: float


@dataclass
class TrailingStopConfig:
    enabled: bool
    activation_rr: float
    trail_atr_multiplier: float


@dataclass
class BotConfig:
    pair: str
    allow_short: bool
    risk: RiskManagementConfig
    sizing: PositionSizingConfig
    ema: EMAConfig
    rsi: RSIConfig
    macd: MACDConfig
    atr: ATRConfig
    volume: VolumeConfig
    entry_long: EntryLongConfig
    stop_loss: StopLossConfig
    take_profit: TakeProfitConfig
    trailing_stop: TrailingStopConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotConfig":
        return cls(
            pair=data["bot"]["trading_pair"],
            allow_short=data["bot"].get("allow_short", False),
            risk=RiskManagementConfig(**data["risk_management"]),
            sizing=PositionSizingConfig(**data["position_sizing"]),
            ema=EMAConfig(**data["indicators"]["ema"]),
            rsi=RSIConfig(**data["indicators"]["rsi"]),
            macd=MACDConfig(**data["indicators"]["macd"]),
            atr=ATRConfig(**data["indicators"]["atr"]),
            volume=VolumeConfig(**data["indicators"]["volume"]),
            entry_long=EntryLongConfig(**data["entry_rules"]["long"]),
            stop_loss=StopLossConfig(**data["exit_rules"]["stop_loss"]),
            take_profit=TakeProfitConfig(**data["exit_rules"]["take_profit"]),
            trailing_stop=TrailingStopConfig(**data["exit_rules"]["trailing_stop"]),
        )


@dataclass
class MarketSnapshot:
    timestamp: datetime
    close: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    volume: float
    volume_ma: float
    spread_percent: float
    data_ok: bool = True
    major_news: bool = False


@dataclass
class RiskState:
    equity: float
    daily_drawdown_percent: float
    total_drawdown_percent: float
    last_loss_timestamp: Optional[datetime] = None


@dataclass
class TradePlan:
    action: str
    reason: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity_btc: Optional[float] = None
    risk_amount: Optional[float] = None
    checks: Optional[List[str]] = None


class DecisionEngine:
    """Confluence-based spot BTC decision engine based on prompt/config constraints."""

    def __init__(self, config: BotConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.log = logger or logging.getLogger("btc_spot_bot")

    def evaluate(self, m: MarketSnapshot, state: RiskState) -> TradePlan:
        checks: List[str] = []

        if not m.data_ok:
            return TradePlan("HOLD", "Data gap or inconsistency detected", checks=["data_ok=false"])

        if state.daily_drawdown_percent >= self.config.risk.max_daily_drawdown_percent:
            return TradePlan("HALT", "Max daily drawdown reached", checks=["daily_dd_limit_hit"])

        if state.total_drawdown_percent >= self.config.risk.max_total_drawdown_percent:
            return TradePlan("HALT", "Max total drawdown reached", checks=["total_dd_limit_hit"])

        if state.last_loss_timestamp is not None:
            cooldown = timedelta(minutes=self.config.risk.cooldown_after_loss_minutes)
            if m.timestamp < state.last_loss_timestamp + cooldown:
                return TradePlan("HOLD", "Cooldown active after recent loss", checks=["cooldown_active"])

        if m.major_news:
            return TradePlan("HOLD", "Major news filter active", checks=["major_news=true"])

        if m.spread_percent > 0.2:
            return TradePlan("HOLD", "Spread too high", checks=[f"spread={m.spread_percent:.3f}"])

        trend_ok = m.close > m.ema_trend and m.ema_fast > m.ema_slow
        checks.append(f"trend_ok={trend_ok}")

        momentum_ok = (m.macd > m.macd_signal) and (self.config.entry_long.rsi_min <= m.rsi <= self.config.entry_long.rsi_max)
        checks.append(f"momentum_ok={momentum_ok}")

        volume_ok = m.volume >= (m.volume_ma * self.config.volume.spike_multiplier)
        checks.append(f"volume_ok={volume_ok}")

        if not (trend_ok and momentum_ok and volume_ok):
            return TradePlan("HOLD", "Confluence conditions not met", checks=checks)

        risk_amount = state.equity * (self.config.risk.risk_per_trade_percent / 100.0)
        stop_distance = m.atr * self.config.stop_loss.atr_multiplier
        if stop_distance <= 0:
            return TradePlan("HOLD", "Invalid ATR stop distance", checks=checks + ["atr_invalid"])

        fee_slippage_factor = 1 + ((self.config.sizing.fee_buffer_percent + self.config.sizing.slippage_buffer_percent) / 100.0)
        quantity = (risk_amount / stop_distance) / fee_slippage_factor

        entry = m.close
        stop = entry - stop_distance
        rr = max(self.config.risk.min_rr_ratio, self.config.take_profit.rr_ratio)
        take = entry + (stop_distance * rr)

        if stop >= entry or take <= entry:
            return TradePlan("HOLD", "Invalid SL/TP placement", checks=checks + ["sl_tp_invalid"])

        return TradePlan(
            action="BUY",
            reason="Long setup confirmed by trend + momentum + volume confluence",
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
            quantity_btc=quantity,
            risk_amount=risk_amount,
            checks=checks,
        )


def load_yaml_config(path: str) -> Dict[str, Any]:
    """YAML loader kept optional to avoid hard dependency during unit tests."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config.yaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = BotConfig.from_dict(load_yaml_config("config.yaml"))
    engine = DecisionEngine(cfg)
    now = datetime.utcnow()

    mock = MarketSnapshot(
        timestamp=now,
        close=65000,
        ema_fast=64800,
        ema_slow=64500,
        ema_trend=64000,
        rsi=55,
        macd=40,
        macd_signal=20,
        atr=400,
        volume=1800,
        volume_ma=1000,
        spread_percent=0.08,
    )
    risk_state = RiskState(equity=10000, daily_drawdown_percent=0.5, total_drawdown_percent=1.0)
    plan = engine.evaluate(mock, risk_state)
    print(plan)
