def IOU(lt1_x, lt1_y, rb1_x, rb1_y, lt2_x, lt2_y, rb2_x, rb2_y):
    """
    Compute the Intersection over Union of two bounding boxes.
    """
    W = min(rb1_x, rb2_x) - max(lt1_x, lt2_x)
    H = min(rb1_y, rb2_y) - max(lt1_y, lt2_y)
    if W <= 0 or H <= 0:
        return 0
    SA = (rb1_x - lt1_x) * (rb1_y - lt1_y)
    SB = (rb2_x - lt2_x) * (rb2_y - lt2_y)
    cross = W * H
    return cross/(SA + SB - cross)


if __name__ == "__main__":
    lt1_x = 1
    lt1_y = 1
    rb1_x = 2
    rb1_y = 2
    lt2_x = 2
    lt2_y = 2
    rb2_x = 3
    rb2_y = 3
    print(IOU(lt1_x, lt1_y, rb1_x, rb1_y, lt2_x, lt2_y, rb2_x, rb2_y))
