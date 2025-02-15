// src/components/TradeGraph.tsx

import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { TradeEvent } from "@/types/types";

interface TradeGraphProps {
  symbol: string;
  trades: TradeEvent[];
}

const TradeGraph: React.FC<TradeGraphProps> = ({ symbol, trades }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    // Set dimensions and margins
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 400 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    // Select the SVG element
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous contents

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Parse the data
    const data = trades
      .filter((trade) => trade.data.symbol === symbol)
      .map((trade) => ({
        timestamp: new Date(trade.data.timestamp * 1000), // Assuming timestamp is in seconds
        price: trade.data.trade_price,
      }));

    if (data.length === 0) return; // No data to display

    // Set up scales
    const x = d3
      .scaleTime()
      .domain(d3.extent(data, (d) => d.timestamp) as [Date, Date])
      .range([0, width]);

    const y = d3
      .scaleLinear()
      .domain([
        d3.min(data, (d) => d.price) as number,
        d3.max(data, (d) => d.price) as number,
      ])
      .nice()
      .range([height, 0]);

    // Add X axis
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

    // Add Y axis
    g.append("g").call(d3.axisLeft(y));

    // Define the line
    const line = d3
      .line<{ timestamp: Date; price: number }>()
      .x((d) => x(d.timestamp))
      .y((d) => y(d.price))
      .curve(d3.curveMonotoneX);

    // Add the line path
    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("d", line);

    // Add points
    g.selectAll("dot")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.timestamp))
      .attr("cy", (d) => y(d.price))
      .attr("r", 3)
      .attr("fill", "steelblue");
  }, [trades, symbol]);

  return <svg ref={svgRef} width={400} height={200}></svg>;
};

export default TradeGraph;
