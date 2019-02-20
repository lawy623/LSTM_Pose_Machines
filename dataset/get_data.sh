# get JHMDB
wget http://files.is.tue.mpg.de/jhmdb/joint_positions.zip
wget http://files.is.tue.mpg.de/jhmdb/puppet_mask.zip   #We need to mask to infer the bounding box.
wget http://files.is.tue.mpg.de/jhmdb/sub_splits.zip
wget http://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz

unzip sub_splits.zip -d JHMDB/sub_splits
unzip puppet_mask.zip -d JHMDB/
unzip joint_positions.zip -d JHMDB/
tar -zxvf Rename_Images.tar.gz -C JHMDB/
rm -f Rename_Images.tar.gz
rm -f joint_positions.zip
rm -f sub_splits.zip
rm -f puppet_mask.zip

# get PENN
wget -O Penn_Action.tar.gz https://public.boxcloud.com/d/1/b1!0ObleMBJzNK-uvDftp74Yto2WKLYfsS-Cj0T-HqNm8LqqeKEALQr2KOx45FYfH2_F301z1wcNV5g2ilToMFf5fbLjQ4ubYgMxeEO_WYW3JDm8DMKC7ZS7t-BB63K9nV-FmHdOEOgkL1iwJy7C4qKifzosUK3wVZeiM_0YAuLwN24O28dawxOCZsKJR-7XobML-UdEMmMrlU-AY8hBFYKyrK4425uvmCxnIFU91bS9EQnUeNG_DV_9hrIkvfwSqw2ze4Q9oToS
tar -zxvf Penn_Action.tar.gz -C PENN/
rm -f Penn_Action.tar.gz
