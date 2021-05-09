from itertools import count
from Tools.scripts.dutree import display
import pandas
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
from pyspark.shell import sqlContext, sql
from pyspark.sql import *
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType


import seaborn as sns
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Rationalization of electric energy consumption") \
    .getOrCreate()

def avg(x):
    av=(x*100)/11476
    if av>50:
        return True
    else:
        return False



print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Read data")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
df = spark.read.option("header", "true").csv("Book1.csv")
df.show(n=5)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Change not avalible to null")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
columns = df.columns
for col in columns:
    df = df.withColumn(col, F.when(df[col] == 'Not Available',None).otherwise(df[col]))

df.show(n=5)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Chacking data types")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
var1=df.schema
for i in var1:
    print(i)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Changing data type to float")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
dfnew = df.withColumn("Largest Property Use Type - Gross Floor Area (ft)",df["Largest Property Use Type - Gross Floor Area (ft)"].cast(FloatType()))\
    .withColumn("2nd Largest Property Use - Gross Floor Area (ft)",df["2nd Largest Property Use - Gross Floor Area (ft)"].cast(FloatType()))\
    .withColumn("3rd Largest Property Use Type - Gross Floor Area (ft)",df["3rd Largest Property Use Type - Gross Floor Area (ft)"].cast(FloatType()))\
    .withColumn("Site EUI (kBtu/ft)",df["Site EUI (kBtu/ft)"].cast(FloatType()))\
    .withColumn("Weather Normalized Site EUI (kBtu/ft)",df["Weather Normalized Site EUI (kBtu/ft)"].cast(FloatType()))\
    .withColumn("Weather Normalized Site Electricity Intensity (kWh/ft)",df["Weather Normalized Site Electricity Intensity (kWh/ft)"].cast(FloatType()))\
    .withColumn("Weather Normalized Site Natural Gas Intensity (therms/ft)",df["Weather Normalized Site Natural Gas Intensity (therms/ft)"].cast(FloatType()))\
    .withColumn("Weather Normalized Source EUI (kBtu/ft)",df["Weather Normalized Source EUI (kBtu/ft)"].cast(FloatType()))\
    .withColumn("Water Intensity (All Water Sources) (gal/ft)",df["Water Intensity (All Water Sources) (gal/ft)"].cast(FloatType()))\
    .withColumn("Source EUI (kBtu/ft)",df["Source EUI (kBtu/ft)"].cast(FloatType()))\
    .withColumn("Fuel Oil #1 Use (kBtu)",df["Fuel Oil #1 Use (kBtu)"].cast(FloatType()))\
    .withColumn("Fuel Oil #2 Use (kBtu)",df["Fuel Oil #2 Use (kBtu)"].cast(FloatType()))\
    .withColumn("Fuel Oil #4 Use (kBtu)",df["Fuel Oil #4 Use (kBtu)"].cast(FloatType()))\
    .withColumn("Fuel Oil #5 & 6 Use (kBtu)",df["Fuel Oil #5 & 6 Use (kBtu)"].cast(FloatType()))\
    .withColumn("Diesel #2 Use (kBtu)",df["Diesel #2 Use (kBtu)"].cast(FloatType()))\
    .withColumn("District Steam Use (kBtu)",df["District Steam Use (kBtu)"].cast(FloatType()))\
    .withColumn("Natural Gas Use (kBtu)",df["Natural Gas Use (kBtu)"].cast(FloatType()))\
    .withColumn("Electricity Use - Grid Purchase (kBtu)",df["Electricity Use - Grid Purchase (kBtu)"].cast(FloatType()))\
    .withColumn("Total GHG Emissions (Metric Tons CO2e)",df["Total GHG Emissions (Metric Tons CO2e)"].cast(FloatType()))\
    .withColumn("Direct GHG Emissions (Metric Tons CO2e)",df["Direct GHG Emissions (Metric Tons CO2e)"].cast(FloatType()))\
    .withColumn("Indirect GHG Emissions (Metric Tons CO2e)",df["Indirect GHG Emissions (Metric Tons CO2e)"].cast(FloatType()))\
    .withColumn("Weather Normalized Site Electricity (kWh)",df["Weather Normalized Site Electricity (kWh)"].cast(FloatType()))\
    .withColumn("Weather Normalized Site Natural Gas Use (therms)",df["Weather Normalized Site Natural Gas Use (therms)"].cast(FloatType()))\
    .withColumn("Water Use (All Water Sources) (kgal)",df["Water Use (All Water Sources) (kgal)"].cast(FloatType()))\
    .withColumn("ENERGY STAR Score",df["ENERGY STAR Score"].cast(IntegerType()))\
    .withColumn("Number of Buildings - Self-reported",df["Number of Buildings - Self-reported"].cast(FloatType()))\
    .withColumn("Property GFA - Self-Reported (ft)",df["Property GFA - Self-Reported (ft)"].cast(FloatType()))


var=dfnew.schema
for i in var:
    print(i)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Calculating null percentage")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
columns1=dfnew.columns
list=[]
for co in columns1:
    print("Calculating on progress.....")
    s=dfnew.select([count(when(dfnew[co].isNull(),True))]).collect()[0][0]
    if avg(s):
        list.append(dfnew[co])

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("                             Drop columns")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
delist=["Address 2","2nd Largest Property Use Type","2nd Largest Property Use - Gross Floor Area (ft)","3rd Largest Property Use Type","3rd Largest Property Use Type - Gross Floor Area (ft)","Fuel Oil #1 Use (kBtu)","Fuel Oil #2 Use (kBtu)","Fuel Oil #4 Use (kBtu)","Fuel Oil #5 & 6 Use (kBtu)","Diesel #2 Use (kBtu)","District Steam Use (kBtu)"]
df3=dfnew.drop(*delist)
df.show(n=1)
df3.show(n=1)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

dfsum=df3.groupby("ENERGY STAR Score").sum("Number of Buildings - Self-reported")
x=dfsum.toPandas()["ENERGY STAR Score"].values.tolist()
y=dfsum.toPandas()["sum(Number of Buildings - Self-reported)"].values.tolist()
plt.bar(x,y,label="Energy score distrbution of buldings")
plt.xlabel("Energy Score")
plt.ylabel("# of Buldings")
plt.show()

group_data = df3.groupBy("Primary Property Type - Self Selected")
group_data.agg({'Primary Property Type - Self Selected':'count','ENERGY STAR Score':'sum'}).sort('count(Primary Property Type - Self Selected)',ascending=False).show(truncate=False,n=100)

filtered_by_property_office = df3.filter(F.col('Primary Property Type - Self Selected') == 'Office').select(F.col('ENERGY STAR Score').alias('ENERGY STAR Score_property_office'))
x1=filtered_by_property_office.toPandas()["ENERGY STAR Score_property_office"].values.tolist()
filtered_by_property_Multifamily = df3.filter(F.col('Primary Property Type - Self Selected') == 'Multifamily Housing').select(F.col('ENERGY STAR Score').alias('ENERGY STAR Score_property_Multifamily'))
x2=filtered_by_property_Multifamily.toPandas()["ENERGY STAR Score_property_Multifamily"].values.tolist()
filtered_by_property_NonRefrigeratedWarehouse = df3.filter(F.col('Primary Property Type - Self Selected') == 'Non-Refrigerated Warehouse').select(F.col('ENERGY STAR Score').alias('ENERGY STAR Score_property_Non-Refrigerated Warehouse'))
x3=filtered_by_property_NonRefrigeratedWarehouse.toPandas()["ENERGY STAR Score_property_Non-Refrigerated Warehouse"].values.tolist()

filtered_by_property_Hotel = df3.filter(F.col('Primary Property Type - Self Selected') == 'Hotel').select(F.col('ENERGY STAR Score').alias('ENERGY STAR Score_property_Hotel'))
x5=filtered_by_property_Hotel.toPandas()["ENERGY STAR Score_property_Hotel"].values.tolist()
df4 = df3.toPandas()
fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

colors = ['red']
ax0.hist(x1, 10, density=True, histtype='bar', color=colors, label='office')
ax0.legend(prop={'size': 10})
ax0.set_title('enerji star over office')

colors = ['red']
ax1.hist(x2, 10, density=True, histtype='bar', color=colors, label='Multifamily')
ax1.legend(prop={'size': 10})
ax1.set_title('enerji star over Multifamily')

colors = ['red']
ax2.hist(x3, 10, density=True, histtype='bar', color=colors, label='Non-Refrigerated Warehouse')
ax2.legend(prop={'size': 10})
ax2.set_title('enerji star over Non-Refrigerated Warehouse')

colors = ['red']
ax3.hist(x5, 10, density=True, histtype='bar', color=colors, label='Hotels')
ax3.legend(prop={'size': 10})
ax3.set_title('enerji star over  Hotels')



fig.tight_layout()
plt.show()



correlations_data = df4.corr()['ENERGY STAR Score'].sort_values()
print(correlations_data*10)
print('/////////////////////Hotel (internsity- use - gross floor)///////////////')
property_hotel_ele_internsity = df3.filter(F.col('Primary Property Type - Self Selected') == 'Hotel').select(F.col('Weather Normalized Site Electricity Intensity (kWh/ft)').alias('property_hotel_ele_internsity'))
property_hotel_ele_internsity.sort('property_hotel_ele_internsity',ascending=False)
x5=property_hotel_ele_internsity.toPandas()

property_hotel_gross_floor = df3.filter(F.col('Primary Property Type - Self Selected') == 'Hotel').select(F.col('Largest Property Use Type - Gross Floor Area (ft)').alias('property_hotel_gross_floor'))
property_hotel_gross_floor.sort('property_hotel_gross_floor',ascending=False)
x6=property_hotel_gross_floor.toPandas()["property_hotel_gross_floor"].values.tolist()

property_hotel_Electricity_Use = df3.filter(F.col('Primary Property Type - Self Selected') == 'Hotel').select(F.col('Electricity Use - Grid Purchase (kBtu)').alias('property_hotel_Electricity_Use(kWh)'))
property_hotel_Electricity_Use_df=property_hotel_Electricity_Use.withColumn('property_hotel_Electricity_Use(kWh)',property_hotel_Electricity_Use['property_hotel_Electricity_Use(kWh)']/3412.14)
property_hotel_Electricity_Use_df.sort('property_hotel_Electricity_Use(kWh)',ascending=False).show(n=100)
x7=property_hotel_Electricity_Use_df.toPandas()["property_hotel_Electricity_Use(kWh)"].values.tolist()

compare0=property_hotel_gross_floor.select('property_hotel_gross_floor')
comp0=compare0.join(property_hotel_Electricity_Use_df,how='outer')
comp0.sort("property_hotel_Electricity_Use(kWh)",ascending=False)
x8=comp0.toPandas()


print('/////////////////////// hotel end ////////////////////')
print('------------------------------------------------------')
print('/////////////////////Multifamily (internsity- use - gross floor)///////////////')
property_Multifamily_Housing_Electricity_Intensity = df3.filter(F.col('Primary Property Type - Self Selected') == 'Multifamily Housing').select(F.col('Weather Normalized Site Electricity Intensity (kWh/ft)').alias('property_Multiy_famliy_Electricity_Intensity'))
property_Multifamily_Housing_Electricity_Intensity.sort('property_Multiy_famliy_Electricity_Intensity')
x9=property_Multifamily_Housing_Electricity_Intensity.toPandas()
tt=property_Multifamily_Housing_Electricity_Intensity.toPandas()

property_Multifamily_Housing_gross_floor = df3.filter(F.col('Primary Property Type - Self Selected') == 'Multifamily Housing').select(F.col('Largest Property Use Type - Gross Floor Area (ft)').alias('property_Multiy_famliy_gross_floor'))
property_Multifamily_Housing_gross_floor.sort('property_Multiy_famliy_gross_floor')
x10=property_Multifamily_Housing_gross_floor.toPandas()["property_Multiy_famliy_gross_floor"].values.tolist()

property_Multifamily_Electricity_Use = df3.filter(F.col('Primary Property Type - Self Selected') == 'Multifamily Housing').select(F.col('Electricity Use - Grid Purchase (kBtu)').alias('property_Multiy_famliy_Electricity_Use'))
property_Multifamily_Electricity_Use_df=property_Multifamily_Electricity_Use.withColumn('property_Multifamily_Electricity_Use(kWh)',property_Multifamily_Electricity_Use['property_Multiy_famliy_Electricity_Use']/3412.14)
x11=property_Multifamily_Electricity_Use_df.toPandas()

compar1=property_Multifamily_Housing_gross_floor.select('property_Multiy_famliy_gross_floor')
comp1=compar1.join(property_Multifamily_Electricity_Use_df,how='outer')
comp1.sort('property_Multifamily_Electricity_Use(kWh)')

print('////////////////////////Multiy_famliy end ////////////////////')
print('-----------------------------------------------------------------')
print('/////////////// office (internsity- use - gross floor) /////////////////')

property_office_Electricity_Intensity = df3.filter(F.col('Primary Property Type - Self Selected') == 'Office').select(F.col('Weather Normalized Site Electricity Intensity (kWh/ft)').alias('property_Office_Electricity_Intensity'))
property_office_Electricity_Intensity.sort('property_Office_Electricity_Intensity')
xx=property_office_Electricity_Intensity.toPandas()

gross_floor=df3.filter(F.col('Primary Property Type - Self Selected')=='Office').select(F.col('Largest Property Use Type - Gross Floor Area (ft)').alias('Office_gross_floor'))
gross_floor.sort('Office_gross_floor')
x12=gross_floor.toPandas()


ele_use=df3.filter(F.col('Primary Property Type - Self Selected')=='Office').select(F.col('Electricity Use - Grid Purchase (kBtu)').alias('office_Electricity_Use(kWh)'))
ele_use_df=ele_use.withColumn('office_Electricity_Use(kWh)',ele_use['office_Electricity_Use(kWh)']/3412.14)
ele_use_df.sort('office_Electricity_Use(kWh)').show(n=100)
x13=ele_use_df.toPandas()

compar2=ele_use.select('office_Electricity_Use(kWh)')
comp2=compar2.join(gross_floor,how='outer')
x14=comp2.toPandas()



x14.plot(x='Office_gross_floor',y='office_Electricity_Use(kWh)',kind='scatter')
x8.plot(x='property_hotel_gross_floor',y='property_hotel_Electricity_Use(kWh)',kind="scatter")
plt.show()