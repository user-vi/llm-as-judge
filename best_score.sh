while :; do clear; n=$(grep -n -E "Best score on valset" gepa.log | tail -n1 | cut -d: -f1); [[ $n ]] && sed -n "${n}p" gepa.log  || echo "Pattern not found"; sleep 5; done
