import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

interface TradeData {
  timestamp: number;
  trade_price: number;
}

interface TradeGraphProps {
  trades: TradeData[]; // List of trades
}

const TradeGraph: React.FC<TradeGraphProps> = ({ trades }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const width = 800; // Graph width
  const height = 400; // Graph height
  const margin = { top: 20, right: 30, bottom: 30, left: 50 };

  useEffect(() => {
    if (!svgRef.current) return;

    // Initialize the SVG
    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Set up scales
    const xScale = d3
      .scaleTime()
      .domain(
        d3.extent(trades, (d) => new Date(d.timestamp * 1000)) as [Date, Date]
      )
      .range([margin.left, width - margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(trades, (d) => d.trade_price) || 100]) // Default to 100 if no trades
      .range([height - margin.bottom, margin.top]);

    // Draw Axes
    svg.selectAll(".x-axis").remove();
    svg
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        d3
          .axisBottom(xScale)
          .ticks(5)
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          .tickFormat((d: Date | d3.NumberValue, _i: number) =>
            d3.timeFormat("%H:%M:%S")(d as Date)
          )
      );

    svg.selectAll(".y-axis").remove();
    svg
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Draw Line
    const line = d3
      .line<TradeData>()
      .x((d) => xScale(new Date(d.timestamp * 1000)))
      .y((d) => yScale(d.trade_price))
      .curve(d3.curveMonotoneX);

    svg.selectAll(".trade-line").remove();
    svg
      .append("path")
      .datum(trades)
      .attr("class", "trade-line")
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 2)
      .attr("d", line);

    // Add Circles for Points
    svg.selectAll(".trade-point").remove();
    svg
      .selectAll(".trade-point")
      .data(trades)
      .enter()
      .append("circle")
      .attr("class", "trade-point")
      .attr("cx", (d) => xScale(new Date(d.timestamp * 1000)))
      .attr("cy", (d) => yScale(d.trade_price))
      .attr("r", 4)
      .attr("fill", "red");
  }, [trades]);

  return <svg ref={svgRef}></svg>;
};

export default TradeGraph;
