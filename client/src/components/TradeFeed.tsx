// src/components/TradeFeed.tsx

"use client";

import React, { useMemo } from "react";
import { useTradeContext } from "@/context/TradeContext";
import TradeGraph from "./TradeGraph";

const TradeFeed: React.FC = () => {
  const { trades } = useTradeContext();

  // Get unique symbols from trades
  const symbols = useMemo(() => {
    const uniqueSymbols = new Set<string>();
    trades.forEach((trade) => uniqueSymbols.add(trade.data.symbol));
    return Array.from(uniqueSymbols);
  }, [trades]);

  return (
    <div className="p-4">
      <h2 className="text-lg font-bold mb-4">Trade Feed</h2>
      <div className="space-y-6">
        {symbols.map((symbol) => (
          <div key={symbol}>
            <h3 className="text-md font-semibold mb-2">{symbol}</h3>
            <TradeGraph symbol={symbol} trades={trades} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default TradeFeed;
