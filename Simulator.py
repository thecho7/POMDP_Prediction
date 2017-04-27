import csv
import operator
import time
import numpy as np
import sys
# Parameter Setup

date = time.strftime("%Y%m%d")
ext = 'txt'
num_items = 15
price_matrix = []
moving_average_mat = np.zeros((num_items, num_items))
normalized_correl_mat = np.zeros((num_items, num_items))
item_and_threshold = {0:[0.4, 0], 1:[0.2, 0], 2:[0.2, 0], 3:[0.2, 0], 4:[0.2, 0], 5:[0.2, 0], 6:[0.5, 0], 7:[0.4, 0], 8:[0.4, 0], 9:[0.15, 0], 10:[0.15, 0], 11:[0.2, 0], 12:[0.2, 0], 13:[0.2, 0], 14:[0.5, 0]}

print("Input your opinion value D")
Opinion = input()
moving_average_param = 1e-3 # If moving_average_param is large, a recent tendency is considered strongly.
moving_average_increase = 1e-5 # If moving_average_increase is large, a gradient of recent tendency becomes larger.
correl_ratio_param = 5e-1 # If correl_ratio_param gets larger, the correlation between the items gets smaller.
charge_threshold = 200  # Let's suppose the maximum price moving is about 200. Since the weight summation is always 1, it should be 200.
charge_diminish = 1e-3 # Charge_diminish is a parameter that reduce the charge when the total charge is over than 'charge_threshold'.


# Obtain covariance between the items using moving average
def similarity_moving_average(pricefile, moving_average_param, moving_average_increase):

    day = 0
    for row in pricefile:
        for i in range(len(row)):
            if row[i] != '':
                price_matrix.append(int(row[i]))
            else:
                price_matrix.append(0)
        day = day + 1

    price_mat = np.reshape(price_matrix, (day, num_items))
    print("The number of days : %d " % day)

    for k in range(day - 1):
        for i in range(num_items):
            for j in range(num_items):
                if price_mat[k][j] != 0 and i != j:
                    if k == 0:
                        moving_average_mat[i][j] = (price_mat[k + 1][i] - price_mat[k][i]) - (price_mat[k + 1][j] - price_mat[k][j])
                    else:
                        moving_average_mat[i][j] = ((1 - moving_average_param) * moving_average_mat[i][j]) + (moving_average_param * (price_mat[k + 1][i] - price_mat[k][i]) - (price_mat[k + 1][j] - price_mat[k][j]))
        moving_average_param = moving_average_param + moving_average_increase


    return moving_average_mat, price_mat, day


def normalization(moving_average_mat):
    correl_ratio = []

    for i in range(num_items):
        correl_max = moving_average_mat[i][0]
        correl_min = moving_average_mat[i][0]
        for j in range(1, num_items):
            if moving_average_mat[i][j] > correl_max:
                correl_max = moving_average_mat[i][j]
            elif moving_average_mat[i][j] < correl_min:
                correl_min = moving_average_mat[i][j]
        correl_ratio.append(correl_max - correl_min)

    for i in range(num_items):
        for j in range(num_items):
            normalized_correl_mat[i][j] = moving_average_mat[i][j] / (correl_ratio[i] * correl_ratio_param)

    for i in range(num_items):
        norm_sum = sum(normalized_correl_mat[i])
        for j in range(num_items):
            if i == j:
                normalized_correl_mat[i][j] = 1
            else:
                normalized_correl_mat[i][j] = normalized_correl_mat[i][j] - ((1 + norm_sum) / (num_items - 1))

    return normalized_correl_mat


def pomdp_selection(recordfile, price_mat, normalized_correl_mat, weightfile, day, D, charge_threshold, charge_diminsh):

    today_weight = np.zeros(num_items)
    new_weight = np.zeros(num_items)
    profit_mat = np.zeros(num_items)

    yesterday_price = price_mat[day - 2]
    today_price = price_mat[day - 1]

    # Finding the weight distribution of today
    for row in weightfile:
        for i in range(num_items):
            today_weight[i] = float(row[i])


    for i in range(num_items):
        profit_mat[i] = float(today_weight[i]) * (int(today_price[i]) - int(yesterday_price[i]))

    max_index, max_profit = max(enumerate(profit_mat), key=operator.itemgetter(1))
    min_index, min_profit = min(enumerate(profit_mat), key=operator.itemgetter(1))

    largest_weight_index, smallest_weight_index = finding_maxmin_index(new_weight)
    
    print("A Profit of the Most Lucrative Item is : %f" % max_profit)
    recordfile.write("Best Item %d is : and Profit %f\n" %(max_index, max_profit))
    recordfile.write("Worst Item %d is : and Profit %f\n" %(min_index, min_profit))

    print("The Number of Index : %d\n" % max_index)
    print("ITEM NUMBER : %d" % (max_index + 1))
    print(len(new_weight))
    print(len(today_weight))
    for i in range(num_items):
        if i != max_index:
            new_weight[i] = float(today_weight[i]) + float(normalized_correl_mat[max_index][i]) * D
        else:
            new_weight[max_index] = float(today_weight[max_index]) + D

    weight_after_charge = charge_thresholding(today_price, yesterday_price, today_weight, new_weight, charge_threshold, charge_diminsh)
    weight_after_threshold = weight_thresholding(weight_after_charge, largest_weight_index, smallest_weight_index)
    weight_final = charge_thresholding(today_price, yesterday_price, today_weight, weight_after_threshold, charge_threshold,
                                    charge_diminsh)

    return weight_final


def charge_thresholding(today_price, yesterday_price, today_weight, new_weight, charge_threshold, charge_diminish):
    # If the charge is over the given threshold, do rebalancing

    total_charge = 0
    for i in range(num_items):
        total_charge = total_charge + abs(today_price[i] - yesterday_price[i]) * abs(new_weight[i] - float(today_weight[i]))

    if total_charge > charge_threshold:
        print("Charge violates - Adjusting")
        for j in range(num_items):
            if new_weight[j] > today_weight[j]:
                new_weight[j] = new_weight[j] - charge_diminish
            elif new_weight[j] < today_weight[j]:
                new_weight[j] = new_weight[j] + charge_diminish

    return new_weight


def weight_thresholding(new_weight, l_index, s_index):
    # Alert if any item violates the limit. Furthermore, rebalance the items

    # A code is little bit messy, it will be modified using [], and len

    stable_weight = np.zeros(num_items)

    while (stable_weight == new_weight).all() == False:
    
        stable_weight = new_weight

        for i in range(num_items):
            new_weight, l_index, s_index = weight_single_thresholding(new_weight, i, item_and_threshold[i][0], item_and_threshold[i][1], l_index, s_index)
        
##        if new_weight[0] > 0.4 or new_weight[0] < 0:
##            print("Item 1 Error")
##            if new_weight[0] > 0.4:
##                amount = new_weight[0] - 0.4
##                new_weight[0] = new_weight[0] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##                l_index, s_index = finding_maxmin_index(new_weight)
##
##            elif new_weight[0] < 0:
##                new_weight[max_index] = new_weight[l_index] - new_weight[0]
##                new_weight[0] = 0
##                l_index, s_index = finding_maxmin_index(new_weight)
##
##        if new_weight[1] > 0.2 or new_weight[1] < 0:
##            print("Item 2 Error")
##            if new_weight[1] > 0.4:
##                amount = new_weight[0] - 0.2
##                new_weight[1] = new_weight[1] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##                l_index, s_index = finding_maxmin_index(new_weight)
##            elif new_weight[1] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[1]
##                new_weight[1] = 0
##
##        if new_weight[2] > 0.2 or new_weight[2] < 0:
##            print("Item 3 Error")
##            if new_weight[2] > 0.2:
##                amount = new_weight[2] - 0.2
##                new_weight[2] = new_weight[2] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[2] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[2]
##                new_weight[2] = 0
##
##        if new_weight[3] > 0.2 or new_weight[3] < 0:
##            print("Item 4 Error")
##            if new_weight[3] > 0.2:
##                amount = new_weight[3] - 0.2
##                new_weight[3] = new_weight[3] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[3] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[3]
##                new_weight[3] = 0
##
##        if new_weight[4] > 0.2 or new_weight[4] < 0:
##            print("Item 5 Error")
##            if new_weight[4] > 0.2:
##                amount = new_weight[4] - 0.2
##                new_weight[4] = new_weight[4] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[4] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[4]
##                new_weight[4] = 0
##
##        if new_weight[5] > 0.2 or new_weight[5] < 0:
##            print("Item 6 Error")
##            if new_weight[5] > 0.2:
##                amount = new_weight[5] - 0.2
##                new_weight[5] = new_weight[5] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[5] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[5]
##                new_weight[5] = 0
##
##        if new_weight[6] > 0.5 or new_weight[6] < 0:
##            print("Item 7 Error")
##            if new_weight[6] > 0.5:
##                amount = new_weight[6] - 0.5
##                new_weight[6] = new_weight[6] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[6] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[6]
##                new_weight[6] = 0
##
##        if new_weight[7] > 0.4 or new_weight[7] < 0:
##            print("Item 8 Error")
##            if new_weight[7] > 0.4:
##                amount = new_weight[7] - 0.4
##                new_weight[7] = new_weight[7] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[7] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[7]
##                new_weight[7] = 0
##
##        if new_weight[8] > 0.4 or new_weight[8] < 0:
##            print("Item 9 Error")
##            if new_weight[8] > 0.4:
##                amount = new_weight[8] - 0.4
##                new_weight[8] = new_weight[8] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[8] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[8]
##                new_weight[8] = 0
##
##        if new_weight[9] > 0.15 or new_weight[9] < 0:
##            print("Item 10 Error")
##            if new_weight[9] > 0.15:
##                amount = new_weight[9] - 0.15
##                new_weight[9] = new_weight[9] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[9] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[9]
##                new_weight[9] = 0
##
##        if new_weight[10] > 0.15 or new_weight[10] < 0:
##            print("Item 11 Error")
##            if new_weight[10] > 0.15:
##                amount = new_weight[0] - 0.15
##                new_weight[10] = new_weight[10] - amount
##                new_weight[s_index] =  new_weight[s_index] + amount
##            elif new_weight[10] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[10]
##                new_weight[10] = 0
##
##        if new_weight[11] > 0.2 or new_weight[11] < 0:
##            print("Item 12 Error")
##            if new_weight[11] > 0.2:
##                amount = new_weight[11] - 0.2
##                new_weight[11] = new_weight[11] - amount
##                new_weight[s_index] =  new_weight[s_index] + amount
##            elif new_weight[11] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[11]
##                new_weight[11] = 0
##
##        if new_weight[12] > 0.2 or new_weight[12] < 0:
##            print("Item 13 Error")
##            if new_weight[12] > 0.2:
##                amount = new_weight[0] - 0.2
##                new_weight[12] = new_weight[12] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[12] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[12]
##                new_weight[12] = 0
##
##        if new_weight[13] > 0.2 or new_weight[13] < 0:
##            print("Item 14 Error")
##            if new_weight[13] > 0.2:
##                amount = new_weight[0] - 0.2
##                new_weight[13] = new_weight[13] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[13] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[13]
##                new_weight[13] = 0
##
##        if new_weight[14] > 0.5 or new_weight[14] < 0:
##            print("Item 15 Error")
##            if new_weight[14] > 0.5:
##                amount = new_weight[14] - 0.5
##                new_weight[14] = new_weight[14] - amount
##                new_weight[s_index] = new_weight[s_index] + amount
##            elif new_weight[14] < 0:
##                new_weight[l_index] = new_weight[l_index] - new_weight[14]
##                new_weight[14] = 0

        if new_weight[0] + new_weight[1] > 0.4 or new_weight[0] + new_weight[1] < 0.1:
            print("Domestic Stock Error")
            if new_weight[0] + new_weight[1] > 0.4:
                amount = new_weight[0] + new_weight[1] - 0.4
                if new_weight[0] == max(new_weight[0], new_weight[1]):
                    new_weight[0] = new_weight[0] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                else:
                    new_weight[1] = new_weight[1] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

            elif new_weight[0] + new_weight[1] < 0.1:
                amount = 0.1 - (new_weight[0] + new_weight[1])
                if new_weight[0] == min(new_weight[0], new_weight[1]):
                    new_weight[0] = new_weight[0] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                else:
                    new_weight[1] =  new_weight[1] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

        if new_weight[2] + new_weight[3] + new_weight[4] + new_weight[5] > 0.4 or new_weight[2] + new_weight[3] + \
                new_weight[4] + new_weight[5] < 0.1:
            print("Oversea Stock Error")
            if new_weight[2] + new_weight[3] + new_weight[4] + new_weight[5] > 0.4:
                amount = new_weight[2] + new_weight[3] + new_weight[4] + new_weight[5] - 0.4
                if new_weight[2] == max(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[2] =  new_weight[2] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[3] == max(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[3] = new_weight[3] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[4] == max(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[4] =  new_weight[4] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[5] == max(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[5] = new_weight[5] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

            elif new_weight[2] + new_weight[3] + new_weight[4] + new_weight[5] < 0.1:
                amount = 0.1 - (new_weight[2] + new_weight[3] + new_weight[4] + new_weight[5])
                if new_weight[2] == min(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[2] = new_weight[2] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[3] == min(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[3] = new_weight[3] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[4] == min(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[4] = new_weight[4] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[5] == min(new_weight[2], new_weight[3], new_weight[4], new_weight[5]):
                    new_weight[5] = new_weight[5] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)

        if new_weight[6] + new_weight[7] + new_weight[8] > 0.6 or new_weight[6] + new_weight[7] + new_weight[8] < 0.2:
            print("Bond Error")
            if new_weight[6] + new_weight[7] + new_weight[8] > 0.6:
                amount = new_weight[6] + new_weight[7] + new_weight[8] - 0.6
                if new_weight[6] == max(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[6] = new_weight[6] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[7] == max(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[7] = new_weight[7] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[8] == max(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[8] = new_weight[8] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

            elif new_weight[6] + new_weight[7] + new_weight[8] < 0.2:
                amount = 0.2 - (new_weight[6] + new_weight[7] + new_weight[8])
                if new_weight[6] == min(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[6] = new_weight[6] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[7] == min(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[7] = new_weight[7] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[8] == min(new_weight[6], new_weight[7], new_weight[8]):
                    new_weight[8] =  new_weight[8] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                    
        if new_weight[9] + new_weight[10] > 0.2 or new_weight[9] + new_weight[10] < 0.05:
            print("Raw Material Error")
            if new_weight[9] + new_weight[10] > 0.2:
                amount = new_weight[9] + new_weight[10] - 0.2
                if new_weight[9] == max(new_weight[9], new_weight[10]):
                    new_weight[9] = new_weight[9] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[10] == max(new_weight[9], new_weight[10]):
                    new_weight[10] = new_weight[10] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

            elif new_weight[9] + new_weight[10] < 0.2:
                amount = 0.05 - (new_weight[9] + new_weight[10])
                if new_weight[9] == min(new_weight[9], new_weight[10]):
                    new_weight[9] = new_weight[9] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                elif new_weight[10] == min(new_weight[9], new_weight[10]):
                    new_weight[10] =  new_weight[10] + amount
                    l_index, s_index = finding_maxmin_index(new_weight)

        if new_weight[12] + new_weight[13] > 0.2:
            print("FX Error")
            if new_weight[12] + new_weight[13] > 0.4:
                amount = new_weight[12] + new_weight[13] - 0.4
                if new_weight[12] == max(new_weight[12], new_weight[13]):
                    new_weight[12] = new_weight[12] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)
                else:
                    new_weight[13] = new_weight[13] - amount
                    l_index, s_index = finding_maxmin_index(new_weight)

    return new_weight

def weight_single_thresholding(new_weight, item_index, max_thresh, min_thresh, l_index, s_index):
    if new_weight[item_index] > max_thresh or new_weight[item_index] < min_thresh:
        print("Item %d Error" %item_index)
        if new_weight[item_index] > max_thresh:
            amount = new_weight[item_index] - max_thresh
            new_weight[item_index] = new_weight[item_index] - amount
            new_weight[s_index] = new_weight[s_index] + amount
            l_index, s_index = finding_maxmin_index(new_weight)
        elif new_weight[item_index] < min_thresh:
            new_weight[l_index] = new_weight[l_index] - new_weight[item_index]
            new_weight[item_index] = min_thresh
            l_index, s_index = finding_maxmin_index(new_weight)

    return new_weight, l_index, s_index

def finding_maxmin_index(new_weight):
    largest_weight_index, largest_weight = max(enumerate(new_weight), key=operator.itemgetter(1))
    smallest_weight_index, smallest_weight = min(enumerate(new_weight), key=operator.itemgetter(1))

    return largest_weight_index, smallest_weight_index

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

def main():

    tic()

    # Open Files
    pricefile_open = open('./price.csv', 'r')
    weightfile_open = open('./weight.csv', 'a+')
    recordfile = open("./%s.%s" %(date, ext), 'w')

    # Initialization
    pricefile = csv.reader(pricefile_open)
    weightfile = csv.reader(weightfile_open)
    weightfile2 = csv.writer(weightfile_open)

    # Algorithm proceeding
    moving_average_mat, price_mat, day = similarity_moving_average(pricefile, moving_average_param,
                                                                   moving_average_increase)
    normalized_correl_mat = normalization(moving_average_mat)
    new_weight = pomdp_selection(recordfile, price_mat, normalized_correl_mat, weightfile, day, Opinion, charge_threshold,
                                 charge_diminish)

    # File update
    weightfile2.writerow(new_weight)
    recordfile.write("Opinion Value D is : %f\n" %Opinion)
    recordfile.write("moving_average_param is : %f\n" %moving_average_param)
    recordfile.write("moving_average_increase is : %f\n" %moving_average_increase)
    recordfile.write("correl_ratio_param is : %f\n" %correl_ratio_param)
    recordfile.write("charge_threshold is : %d\n" %charge_threshold)
    recordfile.write("charge_diminish is : %f\n" %charge_diminish)

    time.sleep(1)

    # File close
    weightfile_open.close()
    pricefile_open.close()
    recordfile.close()

    toc()

if __name__ == '__main__':
    main()
    sys.exit(0)
