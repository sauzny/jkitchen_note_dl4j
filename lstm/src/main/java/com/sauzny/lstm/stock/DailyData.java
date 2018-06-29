package com.sauzny.lstm.stock;

import lombok.Data;

@Data
public class DailyData {
    //开盘价  
    private double openPrice;  
    //收盘价  
    private double closeprice;  
    //最高价  
    private double maxPrice;  
    //最低价  
    private double minPrice;  
    //成交量  
    private double turnover;  
    //成交额  
    private double volume;  
    
    @Override  
    public String toString(){  
        StringBuilder builder = new StringBuilder();  
        builder.append("开盘价="+this.openPrice+", ");  
        builder.append("收盘价="+this.closeprice+", ");  
        builder.append("最高价="+this.maxPrice+", ");  
        builder.append("最低价="+this.minPrice+", ");  
        builder.append("成交量="+this.turnover+", ");  
        builder.append("成交额="+this.volume);  
        return builder.toString();  
    } 
}
