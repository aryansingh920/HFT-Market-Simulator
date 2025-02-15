// src/context/TradeContext.tsx

"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import { TradeEvent } from "@/types/types";

interface TradeContextType {
  trades: TradeEvent[];
  addTrade: (trade: TradeEvent) => void;
}

const TradeContext = createContext<TradeContextType | undefined>(undefined);

export const TradeProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [trades, setTrades] = useState<TradeEvent[]>([]);

  const addTrade = (trade: TradeEvent) => {
    setTrades((prevTrades) => [...prevTrades, trade]);
  };

  return (
    <TradeContext.Provider value={{ trades, addTrade }}>
      {children}
    </TradeContext.Provider>
  );
};

export const useTradeContext = (): TradeContextType => {
  const context = useContext(TradeContext);
  if (!context) {
    throw new Error("useTradeContext must be used within a TradeProvider");
  }
  return context;
};
